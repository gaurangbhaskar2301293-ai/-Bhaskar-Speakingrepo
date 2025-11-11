async function assess() {
  const transcript = document.getElementById("transcript").value;
  const resp = await fetch("http://localhost:8000/api/assess", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({user_id:1, transcript: transcript, audio_features:{phoneme_error_rate:0.1, speech_rate_z:0}})
  });
  const data = await resp.json();
  document.getElementById("output").textContent = JSON.stringify(data, null, 2);
}
document.getElementById("assessBtn").addEventListener("click", assess);
