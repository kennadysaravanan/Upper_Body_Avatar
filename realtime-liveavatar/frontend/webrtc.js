// WebRTC peer: receives avatar audio+video from the server.
class AvatarPeer {
  constructor(signaling, videoEl) {
    this.sig = signaling;
    this.videoEl = videoEl;
    this.pc = null;
    this.stream = new MediaStream();
  }

  async start(iceServers) {
    this.pc = new RTCPeerConnection({ iceServers });

    this.pc.ontrack = (evt) => {
      this.stream.addTrack(evt.track);
      this.videoEl.srcObject = this.stream;
    };

    this.pc.onicecandidate = (evt) => {
      if (evt.candidate) {
        this.sig.send({
          type: "ice",
          candidate: {
            candidate: evt.candidate.candidate,
            sdpMid: evt.candidate.sdpMid,
            sdpMLineIndex: evt.candidate.sdpMLineIndex,
          },
        });
      }
    };

    this.pc.onconnectionstatechange = () =>
      console.log("pc state:", this.pc.connectionState);

    // we only receive media from the server
    this.pc.addTransceiver("video", { direction: "recvonly" });
    this.pc.addTransceiver("audio", { direction: "recvonly" });

    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    this.sig.send({ type: "offer", sdp: offer.sdp, sdp_type: offer.type });
  }

  async onAnswer(msg) {
    await this.pc.setRemoteDescription({ type: msg.sdp_type, sdp: msg.sdp });
  }

  async onIce(msg) {
    try { await this.pc.addIceCandidate(msg.candidate); }
    catch (e) { console.warn("addIceCandidate failed", e); }
  }

  close() { if (this.pc) this.pc.close(); }
}
window.AvatarPeer = AvatarPeer;
