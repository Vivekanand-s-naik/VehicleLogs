"""
Unit Tests for ntfy.sh Mobile Notifications & Clip Photo Attachments
====================================================================
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.services.notification_service import (
    NotificationService,
    NotificationItem,
    build_notification_photo,
)


def test_notification_text_formatting_and_dispatch():
    service = NotificationService()

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        item = NotificationItem(
            plate_number="KA01AB1234",
            event_type="IN",
            timestamp_str="18-Nov-2026 10:30:00 UTC",
            confidence=0.96,
            state_name="Karnataka",
        )

        success = service._send_http(item)
        assert success is True
        assert mock_post.called

        args, kwargs = mock_post.call_args
        assert "ntfy.sh" in args[0]
        assert kwargs["headers"]["Title"] == "Vehicle Entry: KA01AB1234"
        assert "KA01AB1234" in kwargs["data"].decode("utf-8")


def test_notification_with_clip_photo_attachment():
    service = NotificationService()

    veh_img = np.full((480, 800, 3), (50, 60, 70), dtype=np.uint8)
    plate_img = np.full((60, 220, 3), (255, 255, 255), dtype=np.uint8)
    photo_bytes = build_notification_photo(veh_img, plate_img, "KA01AB1234", "IN", 0.96)
    assert photo_bytes is not None
    assert photo_bytes[:2] == b"\xff\xd8"  # Valid JPEG header

    with patch("requests.put") as mock_put:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_put.return_value = mock_resp

        item = NotificationItem(
            plate_number="KA01AB1234",
            event_type="IN",
            timestamp_str="19-Nov-2026 12:30:00 UTC",
            confidence=0.96,
            state_name="Karnataka",
            image_bytes=photo_bytes,
            clip_path="captures/events/KA01AB1234_20261119_123000.mp4",
        )

        success = service._send_http(item)
        assert success is True
        assert mock_put.called

        args, kwargs = mock_put.call_args
        assert "ntfy.sh" in args[0]
        assert kwargs["data"] == photo_bytes
        assert kwargs["headers"]["Filename"] == "KA01AB1234_IN.jpg"
        assert "KA01AB1234" in kwargs["headers"]["Message"]
        assert "KA01AB1234_20261119_123000.mp4" in kwargs["headers"]["Message"]
        assert "\n" not in kwargs["headers"]["Message"]


def test_notification_photo_fallback_to_text_on_413():
    service = NotificationService()

    with patch("requests.put") as mock_put, patch("requests.post") as mock_post:
        put_resp = MagicMock()
        put_resp.status_code = 413
        put_resp.text = "Payload Too Large"
        mock_put.return_value = put_resp

        post_resp = MagicMock()
        post_resp.status_code = 200
        mock_post.return_value = post_resp

        item = NotificationItem(
            plate_number="MH12DE4567",
            event_type="OUT",
            timestamp_str="19-Nov-2026 12:30:00 UTC",
            confidence=0.92,
            image_bytes=b"\xff\xd8dummy",
        )

        success = service._send_http(item)
        assert success is True
        assert mock_put.called
        assert mock_post.called


def test_notification_failure_does_not_crash():
    service = NotificationService()

    with patch("requests.post", side_effect=Exception("Connection refused")):
        item = NotificationItem(
            plate_number="KA01AB1234",
            event_type="OUT",
            timestamp_str="18-Nov-2026 10:30:00 UTC",
            confidence=0.96,
        )
        success = service._send_http(item)
        assert success is False
