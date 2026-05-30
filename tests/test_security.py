"""
Tests for the security module: DICOM anonymization, encryption, audit logging.

Covers:
  - Hash determinism and one-way property
  - Encryption/decryption round-trip
  - Audit log format and content
  - No PHI in sanitized filenames
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.security.privacy import (
    AuditLogger,
    DataEncryptor,
    DicomAnonymizer,
    create_security_pipeline,
)


# ============================================================
# DICOM Anonymizer tests
# ============================================================

class TestDicomAnonymizer:
    """Tests for DICOM anonymization logic."""

    def test_hash_deterministic(self):
        """Same input + same salt → same hash."""
        anon = DicomAnonymizer(hash_salt="test-salt")
        h1 = anon._hash_value("1.2.3.4.5")
        h2 = anon._hash_value("1.2.3.4.5")
        assert h1 == h2

    def test_hash_different_inputs(self):
        """Different inputs → different hashes."""
        anon = DicomAnonymizer(hash_salt="test-salt")
        h1 = anon._hash_value("1.2.3.4.5")
        h2 = anon._hash_value("5.4.3.2.1")
        assert h1 != h2

    def test_hash_different_salts(self):
        """Same input + different salt → different hash (prevents rainbow tables)."""
        anon1 = DicomAnonymizer(hash_salt="salt-A")
        anon2 = DicomAnonymizer(hash_salt="salt-B")
        h1 = anon1._hash_value("1.2.3.4.5")
        h2 = anon2._hash_value("1.2.3.4.5")
        assert h1 != h2

    def test_hash_dicom_uid_format(self):
        """Hashed values should start with '2.25.' (DICOM UID format)."""
        anon = DicomAnonymizer(hash_salt="test-salt")
        h = anon._hash_value("some-uid")
        assert h.startswith("2.25.")

    def test_default_tags_configured(self):
        """Default remove and hash tags should be populated."""
        anon = DicomAnonymizer()
        assert len(anon.remove_tags) > 0
        assert "PatientName" in anon.remove_tags
        assert len(anon.hash_tags) > 0
        assert "StudyInstanceUID" in anon.hash_tags

    def test_custom_config(self):
        """Config should override default tags."""
        cfg = {
            "fields_to_remove": ["PatientName"],
            "fields_to_hash": ["StudyInstanceUID"],
        }
        anon = DicomAnonymizer(config=cfg)
        assert anon.remove_tags == ["PatientName"]
        assert anon.hash_tags == ["StudyInstanceUID"]


# ============================================================
# Encryption tests
# ============================================================

class TestDataEncryptor:
    """Tests for AES-256-GCM encryption."""

    def test_encrypt_decrypt_roundtrip_bytes(self):
        """Encrypting then decrypting should return original data."""
        enc = DataEncryptor(password="test-password-123")
        original = b"Hello, this is secret medical data!"

        encrypted = enc.encrypt_bytes(original)
        decrypted = enc.decrypt_bytes(encrypted)

        assert decrypted == original

    def test_encrypted_differs_from_plaintext(self):
        """Ciphertext should not contain the plaintext."""
        enc = DataEncryptor(password="test-password-123")
        original = b"Sensitive patient information"

        encrypted = enc.encrypt_bytes(original)
        assert original not in encrypted

    def test_encrypt_decrypt_roundtrip_file(self, tmp_path):
        """File encryption round-trip should preserve content."""
        enc = DataEncryptor(password="file-test-key")

        # Create a test file
        original_data = b"Model weights: " + os.urandom(256)
        input_file = tmp_path / "model.pth"
        encrypted_file = tmp_path / "model.pth.enc"
        decrypted_file = tmp_path / "model_decrypted.pth"

        input_file.write_bytes(original_data)

        enc.encrypt_file(str(input_file), str(encrypted_file))
        enc.decrypt_file(str(encrypted_file), str(decrypted_file))

        assert decrypted_file.read_bytes() == original_data

    def test_wrong_password_fails(self):
        """Decryption with wrong password should raise an error."""
        enc1 = DataEncryptor(password="correct-password")
        enc2 = DataEncryptor(password="wrong-password")

        encrypted = enc1.encrypt_bytes(b"secret data")

        with pytest.raises(Exception):
            enc2.decrypt_bytes(encrypted)

    def test_no_password_raises(self):
        """Missing password should raise ValueError."""
        # Clear env var if set
        env_backup = os.environ.get("ENCRYPTION_KEY")
        os.environ.pop("ENCRYPTION_KEY", None)

        try:
            with pytest.raises(ValueError, match="Encryption password required"):
                DataEncryptor(password="")
        finally:
            if env_backup is not None:
                os.environ["ENCRYPTION_KEY"] = env_backup

    def test_different_encryptions_differ(self):
        """Same plaintext encrypted twice should produce different ciphertext (random nonce)."""
        enc = DataEncryptor(password="test-key")
        original = b"same data"

        e1 = enc.encrypt_bytes(original)
        e2 = enc.encrypt_bytes(original)

        assert e1 != e2  # Different nonces → different ciphertext

    def test_tamper_detection(self):
        """Modifying ciphertext should fail decryption (GCM auth tag)."""
        enc = DataEncryptor(password="integrity-test")
        encrypted = enc.encrypt_bytes(b"important data")

        # Tamper with the ciphertext (flip a byte near the end)
        tampered = bytearray(encrypted)
        tampered[-5] ^= 0xFF
        tampered = bytes(tampered)

        with pytest.raises(Exception):
            enc.decrypt_bytes(tampered)


# ============================================================
# Audit logger tests
# ============================================================

class TestAuditLogger:
    """Tests for HIPAA-compliant audit logging."""

    def test_log_prediction(self, tmp_path):
        """Prediction logs should contain required fields."""
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))

        logger.log_prediction(
            filename="scan_001.dcm",
            prediction="glioma",
            confidence=0.94,
            processing_time_ms=45.2,
        )

        # Read and parse the log
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["action"] == "predict"
        assert event["result"]["prediction"] == "glioma"
        assert event["result"]["confidence"] == 0.94
        assert "timestamp" in event
        assert "event_id" in event

    def test_filename_sanitized(self, tmp_path):
        """Patient-identifying filenames should be hashed in logs."""
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))

        logger.log_prediction(
            filename="JohnDoe_MRI_brain.dcm",
            prediction="meningioma",
            confidence=0.87,
        )

        event = json.loads(log_file.read_text().strip())
        resource = event["resource"]

        # Should NOT contain the patient name
        assert "JohnDoe" not in resource
        # Should keep the extension
        assert resource.endswith(".dcm")
        # Should be a hash
        assert len(resource) > 5

    def test_log_data_access(self, tmp_path):
        """Data access events should be logged."""
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))

        logger.log_data_access(resource="training_data/batch_1.zip", action="read")

        event = json.loads(log_file.read_text().strip())
        assert event["action"] == "data_read"

    def test_log_anonymization(self, tmp_path):
        """Anonymization events should track tags processed."""
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))

        logger.log_anonymization(
            input_file="scan.dcm",
            output_file="scan_anon.dcm",
            tags_removed=15,
            tags_hashed=3,
        )

        event = json.loads(log_file.read_text().strip())
        assert event["action"] == "anonymize"
        assert event["result"]["tags_removed"] == 15
        assert event["result"]["tags_hashed"] == 3

    def test_log_error(self, tmp_path):
        """Error events should include severity."""
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))

        logger.log_error(
            action="predict",
            resource="corrupt.jpg",
            error="Cannot decode image",
        )

        event = json.loads(log_file.read_text().strip())
        assert event["severity"] == "error"
        assert "Cannot decode image" in event["error"]

    def test_append_only(self, tmp_path):
        """Multiple events should append, not overwrite."""
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))

        logger.log_prediction("a.dcm", "glioma", 0.9)
        logger.log_prediction("b.dcm", "meningioma", 0.8)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_get_recent_events(self, tmp_path):
        """get_recent_events should return parsed events."""
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))

        for i in range(5):
            logger.log_prediction(f"scan_{i}.dcm", "pituitary", 0.85)

        events = logger.get_recent_events(n=3)
        assert len(events) == 3
        assert all(e["action"] == "predict" for e in events)


# ============================================================
# Security pipeline factory
# ============================================================

class TestSecurityPipeline:
    """Tests for the convenience factory function."""

    def test_create_all_components(self):
        """create_security_pipeline should return all three components."""
        os.environ["ENCRYPTION_KEY"] = "test-key-for-ci"
        try:
            components = create_security_pipeline()
            assert "anonymizer" in components
            assert "encryptor" in components
            assert "audit_logger" in components
            assert isinstance(components["anonymizer"], DicomAnonymizer)
            assert isinstance(components["encryptor"], DataEncryptor)
            assert isinstance(components["audit_logger"], AuditLogger)
        finally:
            os.environ.pop("ENCRYPTION_KEY", None)
