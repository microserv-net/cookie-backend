
import tempfile, pathlib, pytest
from cookie_backend.auth import DeviceStore, PairingError

def store():
    d = tempfile.mkdtemp()
    return DeviceStore(pathlib.Path(d) / "devices.json")

def test_pairing_round_trip():
    s = store()
    code = s.begin_pairing()
    token = s.complete_pairing(code, "laptop")
    assert s.authenticate(token).name == "laptop"
    assert s.authenticate("nonsense") is None

def test_codes_are_single_use():
    s = store()
    code = s.begin_pairing()
    s.complete_pairing(code, "laptop")
    with pytest.raises(PairingError):
        s.complete_pairing(code, "another")

def test_expired_codes_are_refused():
    s = store()
    code = s.begin_pairing(ttl=-1)
    with pytest.raises(PairingError):
        s.complete_pairing(code, "laptop")

def test_tokens_are_not_stored_in_plaintext():
    s = store()
    token = s.complete_pairing(s.begin_pairing(), "laptop")
    assert token not in s.path.read_text()

def test_revocation_and_persistence():
    s = store()
    token = s.complete_pairing(s.begin_pairing(), "laptop")
    reloaded = DeviceStore(s.path)
    assert reloaded.authenticate(token) is not None
    assert reloaded.revoke("laptop")
    assert DeviceStore(s.path).authenticate(token) is None
