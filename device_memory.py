"""
device_memory.py  –  On-device appliance re-identification + naming layer.

This is the "remember this device" engine that sits on top of the MoE demo.

How it works
------------
When the user names a detected appliance, we crop its bounding box, push the
crop through the *already-loaded* ResNet18 router backbone to get a 512-d
appearance fingerprint, and store {fingerprint -> name} in a small JSON file.

On every later frame, each detected box is fingerprinted the same way and
compared (cosine similarity) against the saved gallery. A close-enough match
means "this is the same appliance you named before", so we show the user's
name instead of the model's generic label — and it survives app restarts
because the gallery lives on disk.

In the real SmartThings integration the `name` field becomes the SmartThings
device id / settings deep-link. The recognition + persistence is identical;
only the action taken on a match changes.

Note on limits: this matches by *appearance*, not by tracking. Two visually
identical units would share an identity. For a single-room demo that's fine,
and appearance-matching is the only thing that survives the camera panning
away and coming back (a tracker loses identity the moment the object leaves
the frame).
"""

import json
import time
import uuid
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  Embedder: turns a cropped appliance image into a normalized 512-d vector.
#  Re-uses the router's ResNet18 backbone so nothing extra has to be loaded.
# --------------------------------------------------------------------------- #
def build_embedder(router_module, device='cpu', use_fp16=False):
    """Return an `embed(pil_crop) -> np.ndarray (512,)` function.

    `router_module` is the full ResNet18 (with the 3-class fc head). We strip
    the final fc layer and use everything up to global average pooling, which
    gives a 512-d feature vector. The vector is L2-normalized so that cosine
    similarity is just a dot product.
    """
    # Imported here so this module stays importable (e.g. for the numpy-only
    # gallery tests) even in environments without torch.
    import torch
    import torch.nn as nn
    import torchvision.transforms as T

    backbone = nn.Sequential(*list(router_module.children())[:-1])
    backbone.eval()

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.inference_mode()
    def embed(pil_crop):
        x = tf(pil_crop).unsqueeze(0)
        if use_fp16:
            x = x.half()
        if device != 'cpu':
            x = x.to(device)
        feat = backbone(x).flatten(1)[0].float().cpu().numpy()
        norm = float(np.linalg.norm(feat)) + 1e-8
        return (feat / norm).astype(np.float32)

    return embed


def _l2(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / (float(np.linalg.norm(vec)) + 1e-8)


# --------------------------------------------------------------------------- #
#  DeviceMemory: the persistent gallery of named appliances.
# --------------------------------------------------------------------------- #
class DeviceMemory:
    """A tiny on-disk gallery mapping appearance fingerprints to user names."""

    def __init__(self, store_path='device_memory.json', sim_thresh=0.82):
        self.store_path = Path(store_path)
        self.sim_thresh = float(sim_thresh)
        self.entries = []   # each: {id, name, expert, vec(np.float32), updated}
        self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self):
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for e in data.get('devices', []):
            self.entries.append({
                'id': e.get('id', uuid.uuid4().hex[:8]),
                'name': e['name'],
                'expert': e.get('expert', 'unknown'),
                'vec': np.asarray(e['vec'], dtype=np.float32),
                'updated': e.get('updated', 0.0),
            })

    def _save(self):
        data = {'devices': [{
            'id': e['id'],
            'name': e['name'],
            'expert': e['expert'],
            'vec': e['vec'].tolist(),
            'updated': e['updated'],
        } for e in self.entries]}
        self.store_path.write_text(json.dumps(data, indent=2))

    # ---- core lookups ------------------------------------------------------
    def match(self, vec):
        """Return (entry, score) for the best gallery match above threshold,
        else None. `vec` must be (or will be treated as) L2-normalized."""
        if not self.entries:
            return None
        vec = _l2(vec)
        sims = np.array([float(np.dot(vec, e['vec'])) for e in self.entries])
        best = int(np.argmax(sims))
        if sims[best] >= self.sim_thresh:
            return self.entries[best], float(sims[best])
        return None

    def remember(self, vec, name, expert='unknown'):
        """Name a device. If it visually matches an existing one, update that
        entry (and nudge its fingerprint via a running average) instead of
        creating a duplicate. Persists immediately. Returns the entry."""
        vec = _l2(vec)
        hit = self.match(vec)
        if hit is not None:
            entry, _ = hit
            entry['name'] = name
            entry['expert'] = expert
            entry['vec'] = _l2(0.7 * entry['vec'] + 0.3 * vec)
            entry['updated'] = time.time()
        else:
            entry = {
                'id': uuid.uuid4().hex[:8],
                'name': name,
                'expert': expert,
                'vec': vec,
                'updated': time.time(),
            }
            self.entries.append(entry)
        self._save()
        return entry

    def forget(self, device_id):
        before = len(self.entries)
        self.entries = [e for e in self.entries if e['id'] != device_id]
        if len(self.entries) != before:
            self._save()
        return before - len(self.entries)

    def forget_all(self):
        self.entries = []
        self._save()

    def names(self):
        return [e['name'] for e in self.entries]

    def __len__(self):
        return len(self.entries)
