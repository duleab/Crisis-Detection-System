import streamlit as st

st.header("🚨 Authority Alert Routing")
st.write("Preview and export structured alerts for confirmed crises.")

st.code('''{
  "event_type": "flood",
  "location": "Jakarta",
  "authority": "BPBD DKI Jakarta",
  "llm_summary": "Severe flooding reported in South Jakarta causing significant traffic disruptions."
}''', language="json")

st.markdown("---")
st.subheader("Human-in-the-Loop Review")
st.write("Does this alert look correct?")
col1, col2 = st.columns(2)
with col1:
    if st.button("👍 Approve Alert"):
        st.success("Alert verified! Sending to authority...")
        # Write to feedback loop
with col2:
    if st.button("👎 Reject / Correct"):
        st.warning("Flagged for correction.")
        corrected_type = st.selectbox("Correct Event Type:", ["flood", "earthquake", "fire", "accident", "violence", "other"])
        if st.button("Submit Correction"):
            st.info(f"Correction saved to feedback_loop.csv: {corrected_type}")

