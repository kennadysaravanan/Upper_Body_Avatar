// WebSocket control + signaling channel.
class SignalingClient {
  constructor() {
    this.ws = null;
    this.handlers = {};
  }
  on(type, fn) { this.handlers[type] = fn; }

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws`;
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url);
      this.ws.onopen = () => resolve();
      this.ws.onerror = (e) => reject(e);
      this.ws.onclose = () => this.handlers["close"] && this.handlers["close"]();
      this.ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data);
        const fn = this.handlers[msg.type];
        if (fn) fn(msg);
        else console.warn("unhandled msg", msg);
      };
    });
  }

  send(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }
  close() { if (this.ws) this.ws.close(); }
}
window.SignalingClient = SignalingClient;
