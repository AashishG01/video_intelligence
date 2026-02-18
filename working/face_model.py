from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self):
        print("⏳ Loading AI Models...")
        self.app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(1024, 1024))
        
        active_providers = self.app.models['detection'].session.get_providers()
        print(f"✅ AI System Ready. Active Provider: {active_providers[0]}")

    def extract_faces(self, frame):
        return self.app.get(frame)