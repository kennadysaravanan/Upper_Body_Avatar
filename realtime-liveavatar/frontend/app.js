// App wiring: connect flow + chat.
(() => {
  const $ = (id) => document.getElementById(id);
  let sig = null, peer = null, imageB64 = null, botRow = null;

  $("image").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    imageB64 = await AvatarUI.fileToBase64(file);
    const prev = $("preview");
    prev.src = imageB64; prev.hidden = false;
  });

  $("connect").addEventListener("click", async () => {
    const apiKey = $("apiKey").value.trim();
    if (!imageB64) return alert("Upload an avatar image first.");
    if (!apiKey.startsWith("sk-")) return alert("Enter a valid OpenAI API key.");

    AvatarUI.setStatus("connecting", "connecting…");
    $("connect").disabled = true;

    sig = new SignalingClient();
    peer = new AvatarPeer(sig, $("avatar"));

    sig.on("ready", async (msg) => {
      await peer.start(msg.ice_servers);
    });
    sig.on("answer", (msg) => peer.onAnswer(msg));
    sig.on("ice", (msg) => peer.onIce(msg));
    sig.on("assistant_text", (msg) => {
      if (!botRow) botRow = AvatarUI.addRow("bot", "");
      if (msg.delta) AvatarUI.appendToRow(botRow, msg.delta);
      if (msg.done) botRow = null;
    });
    sig.on("error", (msg) => {
      AvatarUI.setStatus("error", msg.message);
      console.error(msg.message);
    });
    sig.on("close", () => AvatarUI.setStatus("idle", "disconnected"));

    await sig.connect();
    sig.send({
      type: "hello",
      openai_api_key: apiKey,
      llm_model: $("llmModel").value,
      tts_model: "gpt-4o-mini-tts",
      tts_voice: $("ttsVoice").value,
      avatar_image_b64: imageB64,
      prompt: $("prompt").value,
    });

    $("placeholder").style.display = "none";
    AvatarUI.setStatus("live", "live");
    ["text", "send", "interrupt", "disconnect"].forEach((id) => ($(id).disabled = false));
  });

  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("text").value.trim();
    if (!text) return;
    AvatarUI.addRow("user", text);
    sig.send({ type: "user_text", text });
    $("text").value = "";
  });

  $("interrupt").addEventListener("click", () => sig && sig.send({ type: "interrupt" }));

  $("disconnect").addEventListener("click", () => {
    if (peer) peer.close();
    if (sig) sig.close();
    AvatarUI.setStatus("idle", "idle");
    ["text", "send", "interrupt", "disconnect"].forEach((id) => ($(id).disabled = true));
    $("connect").disabled = false;
  });
})();
