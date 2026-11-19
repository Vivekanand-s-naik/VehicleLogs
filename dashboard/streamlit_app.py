"""
ANPR Monitoring Dashboard (Streamlit)
=====================================
Local visual dashboard communicating directly with the FastAPI backend.
Features:
- Overview KPI summary cards & recent movements
- Live visual camera monitoring feed & real-time telemetry HUD
- Vehicle directory, lookup & history tracking
- Event log filtering & image previews
- System health, GPU/CUDA telemetry & application logs
"""

import os
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# Streamlit Page Config
st.set_page_config(
    page_title="ANPR & Vehicle Management",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def fetch_api(endpoint: str, params: dict = None):
    """Safely fetch JSON from FastAPI backend."""
    try:
        url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=4.0)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


# Sidebar Navigation & Status
st.sidebar.title("🚗 ANPR Gate Control")
st.sidebar.caption("Intelligent Vehicle Entry/Exit Management")

# Quick Health Badge in Sidebar
health = fetch_api("/health")
if health and health.get("status") == "HEALTHY":
    st.sidebar.success(f"● System: {health['status']} ({health.get('device_name', 'CPU')})")
elif health:
    st.sidebar.warning(f"▲ System: {health['status']}")
else:
    st.sidebar.error("✖ Backend API Offline (Connecting...)")

menu = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📹 Live Monitoring", "🚘 Vehicle Directory", "📜 Event History", "⚙️ System Health & Logs"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**ntfy.sh Notifications**\n"
    "Subscribe to topic on your phone:\n"
    "`https://ntfy.sh/anpr_vehicle_events_demo`"
)

# ==========================================
# 1. OVERVIEW
# ==========================================
if menu == "📊 Overview":
    st.title("Gate Entry/Exit Overview")
    summary = fetch_api("/api/dashboard/summary")

    if not summary:
        st.warning("Could not reach backend API at " + API_BASE_URL + ". Ensure `run.py` is active.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registered Vehicles", summary["total_registered_vehicles"])
        col2.metric("Vehicles Currently Inside", summary["vehicles_currently_inside"])
        col3.metric("Today's Entries", summary["today_entries"])
        col4.metric("Today's Exits", summary["today_exits"])

        st.markdown("### Recent Vehicle Activity")
        recent = summary.get("recent_events", [])
        if recent:
            df = pd.DataFrame(recent)
            cols_to_display = ["id", "plate_number", "event_type", "event_timestamp", "ocr_confidence"]
            available_cols = [c for c in cols_to_display if c in df.columns]
            clean_df = df[available_cols].copy()
            clean_df.rename(columns={
                "id": "Event ID",
                "plate_number": "License Plate",
                "event_type": "Movement",
                "event_timestamp": "Timestamp",
                "ocr_confidence": "OCR Conf"
            }, inplace=True)
            st.dataframe(clean_df, use_container_width=True)
        else:
            st.info("No vehicle activity recorded yet today.")


# ==========================================
# 2. LIVE MONITORING
# ==========================================
elif menu == "📹 Live Monitoring":
    st.title("Live Camera Monitoring Feed")
    
    col_stream, col_telemetry = st.columns([3, 1])

    with col_stream:
        stream_url = f"{API_BASE_URL}/api/camera/stream"
        st.markdown(
            f"""
            <div style="border: 2px solid #333; border-radius: 8px; overflow: hidden; background: #000;">
                <img src="{stream_url}" width="100%" style="display: block;" onerror="this.onerror=null;this.src='https://placehold.co/1280x720/222/fff?text=Waiting+for+Camera+Stream...';" />
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Annotated real-time video feed showing detected vehicles, tracking IDs, and license plates.")

    with col_telemetry:
        st.subheader("Live Telemetry")
        status_data = fetch_api("/api/system/status")
        if status_data:
            cam = status_data.get("camera", {})
            metrics = status_data.get("pipeline_metrics", {})
            diag = status_data.get("hardware_diagnostics", {})

            st.metric("Camera FPS", f"{cam.get('fps_actual', 0.0)} FPS")
            st.metric("Inference Latency", f"{metrics.get('inference_time_ms', 0.0)} ms")
            st.metric("OCR Latency", f"{metrics.get('ocr_time_ms', 0.0)} ms")
            st.metric("Hardware Acceleration", f"{diag.get('active_device', 'cpu').upper()}")
            
            if "total_vram_mb" in diag:
                st.caption(f"VRAM: {diag.get('allocated_vram_mb', 0):.0f} / {diag.get('total_vram_mb', 0):.0f} MB")

            if st.button("🔄 Reconnect Camera"):
                r = requests.post(f"{API_BASE_URL}/api/camera/reconnect")
                if r.status_code == 200:
                    st.success("Camera reconnected!")
                else:
                    st.error("Reconnect failed.")
        else:
            st.info("Awaiting telemetry...")


# ==========================================
# 3. VEHICLE DIRECTORY
# ==========================================
elif menu == "🚘 Vehicle Directory":
    st.title("Registered Vehicle Management")

    search_query = st.text_input("Search by Registration Number (e.g. KA01AB1234)", "").strip()
    status_filter = st.selectbox("Filter Status", ["All", "IN", "OUT"])

    params = {"skip": 0, "limit": 50}
    if search_query:
        params["search"] = search_query
    if status_filter != "All":
        params["status"] = status_filter

    data = fetch_api("/api/vehicles", params=params)

    if data and data.get("items"):
        df = pd.DataFrame(data["items"])
        display_df = df[["plate_number", "current_status", "total_entries", "total_exits", "last_seen_at"]].copy()
        display_df.rename(columns={
            "plate_number": "Registration Plate",
            "current_status": "Current Status",
            "total_entries": "Total Entries",
            "total_exits": "Total Exits",
            "last_seen_at": "Last Seen"
        }, inplace=True)
        st.dataframe(display_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Vehicle Inspection & History")
        selected_plate = st.selectbox("Select Vehicle Plate to inspect:", df["plate_number"].tolist())

        if selected_plate:
            history = fetch_api(f"/api/vehicles/{selected_plate}/history")
            if history:
                st.markdown(f"**Movement Timeline for {selected_plate}:**")
                for evt in history:
                    with st.expander(f"{evt['event_type']} - {evt['event_timestamp']} (Conf: {evt['ocr_confidence']*100:.1f}%)"):
                        c1, c2 = st.columns(2)
                        with c1:
                            if evt.get("plate_image_path"):
                                img_url = f"{API_BASE_URL}/api/captures/{evt['plate_image_path'].replace('captures/', '')}"
                                st.image(img_url, caption="Cropped Plate Image", use_container_width=True)
                            else:
                                st.write("No plate crop saved.")
                        with c2:
                            if evt.get("vehicle_image_path"):
                                veh_url = f"{API_BASE_URL}/api/captures/{evt['vehicle_image_path'].replace('captures/', '')}"
                                st.image(veh_url, caption="Vehicle Context Image", use_container_width=True)
                            else:
                                st.write("No vehicle image saved.")
    else:
        st.info("No vehicles match the filter criteria.")


# ==========================================
# 4. EVENT HISTORY
# ==========================================
elif menu == "📜 Event History":
    st.title("Vehicle Entry/Exit Historical Logs")

    c1, c2, c3 = st.columns(3)
    with c1:
        plate_filter = st.text_input("Filter by Plate Number", "")
    with c2:
        type_filter = st.selectbox("Filter Movement", ["All", "IN", "OUT"])
    with c3:
        limit_count = st.slider("Max Events", 10, 100, 30)

    params = {"skip": 0, "limit": limit_count}
    if plate_filter:
        params["plate"] = plate_filter.strip()
    if type_filter != "All":
        params["event_type"] = type_filter

    events_data = fetch_api("/api/events", params=params)

    if events_data and events_data.get("items"):
        events = events_data["items"]
        st.write(f"Showing **{len(events)}** of **{events_data['total']}** total records:")

        for ev in events:
            color = "🟢" if ev["event_type"] == "IN" else "🔴"
            with st.container():
                cols = st.columns([1, 2, 2, 2, 3])
                cols[0].markdown(f"### {color} {ev['event_type']}")
                cols[1].markdown(f"**Plate:** `{ev['plate_number']}`\n\n**Confidence:** `{ev['ocr_confidence']*100:.1f}%`")
                cols[2].markdown(f"**Time:** {ev['event_timestamp']}")
                with cols[3]:
                    if ev.get("plate_image_path"):
                        img_url = f"{API_BASE_URL}/api/captures/{ev['plate_image_path'].replace('captures/', '')}"
                        st.image(img_url, width=160)
                with cols[4]:
                    if ev.get("vehicle_image_path"):
                        veh_url = f"{API_BASE_URL}/api/captures/{ev['vehicle_image_path'].replace('captures/', '')}"
                        st.image(veh_url, width=180)
                st.markdown("---")
    else:
        st.info("No historical events match the specified filters.")


# ==========================================
# 5. SYSTEM HEALTH & LOGS
# ==========================================
elif menu == "⚙️ System Health & Logs":
    st.title("System Diagnostics & Application Logs")

    status_data = fetch_api("/api/system/status")

    if status_data:
        h = status_data.get("health", {})
        cam = status_data.get("camera", {})
        diag = status_data.get("hardware_diagnostics", {})
        notif = status_data.get("notifications", {})

        c1, c2, c3 = st.columns(3)
        c1.markdown("### 🖥️ Hardware & Device")
        c1.write(f"**Active Device:** `{diag.get('active_device')}`")
        c1.write(f"**Device Name:** `{diag.get('device_name')}`")
        if "total_vram_mb" in diag:
            c1.write(f"**Total VRAM:** `{diag.get('total_vram_mb')} MB`")
            c1.write(f"**Allocated VRAM:** `{diag.get('allocated_vram_mb')} MB`")

        c2.markdown("### 📷 Camera Status")
        c2.write(f"**Connected:** `{cam.get('is_connected')}`")
        c2.write(f"**Source:** `{cam.get('source')}`")
        c2.write(f"**FPS:** `{cam.get('fps_actual')} FPS`")
        c2.write(f"**Frames Captured:** `{cam.get('total_frames_captured')}`")

        c3.markdown("### 📲 Push Notifications")
        c3.write(f"**Enabled:** `{notif.get('enabled')}`")
        c3.write(f"**Server:** `{notif.get('server')}`")
        c3.write(f"**Topic:** `{notif.get('topic')}`")
        c3.write(f"**Dispatched:** `{notif.get('total_dispatched')}`")
        c3.write(f"**Failed:** `{notif.get('total_failed')}`")

    st.markdown("---")
    st.subheader("Recent Application Logs")
    logs = fetch_api("/api/system/logs", params={"max_lines": 80})
    if logs:
        st.text_area("Log Output", value="\n".join(logs), height=350)
    else:
        st.write("No logs available.")
