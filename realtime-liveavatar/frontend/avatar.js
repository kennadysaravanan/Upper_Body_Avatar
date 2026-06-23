// UI helpers: image encoding + transcript rendering.
const AvatarUI = {
  fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result); // data:...;base64,xxxx
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  },

  setStatus(state, text) {
    const el = document.getElementById("status");
    el.className = `status ${state}`;
    el.textContent = text || state;
  },

  addRow(who, text) {
    const box = document.getElementById("transcript");
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span class="${who}">${who === "user" ? "You" : "Avatar"}:</span> ` +
      text.replace(/</g, "&lt;");
    box.appendChild(row);
    box.scrollTop = box.scrollHeight;
    return row;
  },

  appendToRow(row, text) {
    row.innerHTML += text.replace(/</g, "&lt;");
    const box = document.getElementById("transcript");
    box.scrollTop = box.scrollHeight;
  },
};
window.AvatarUI = AvatarUI;
