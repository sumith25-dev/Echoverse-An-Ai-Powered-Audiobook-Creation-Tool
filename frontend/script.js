document.getElementById('topicForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const topicInput = document.getElementById('topicInput');
  const messageDiv = document.getElementById('message');
  const audioPlayer = document.getElementById('audioPlayer');
  const downloadLinkDiv = document.getElementById('downloadLink');

  audioPlayer.style.display = 'none';
  downloadLinkDiv.innerHTML = '';
  messageDiv.textContent = '';

  const topic = topicInput.value.trim();
  if (!topic) {
    alert("Please enter a topic or prompt.");
    return;
  }

  messageDiv.textContent = "Generating audiobook... Please wait ⏳";

  try {
    const response = await fetch('http://127.0.0.1:5000/generate', {
      method: 'POST',
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ topic })
    });

    if (!response.ok) {
      const err = await response.json();
      messageDiv.textContent = "Error: " + (err.error || "Unknown error occurred.");
      return;
    }

    const blob = await response.blob();
    const audioUrl = URL.createObjectURL(blob);

    audioPlayer.src = audioUrl;
    audioPlayer.style.display = 'block';
    audioPlayer.load();

    downloadLinkDiv.innerHTML = `<a href="${audioUrl}" download="audiobook.mp3" style="font-weight:bold;">⬇ Download Audiobook</a>`;

    messageDiv.textContent = "Audiobook generated successfully!";

  } catch (error) {
    messageDiv.textContent = "Fetch failed: " + error.message;
  }
});


