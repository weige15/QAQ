import hashlib
import io
import zipfile

import torch


def test_torch_serializes_contiguous_qweight_bytes_little_endian():
    qweight = torch.tensor(
        [-2147483648, 0x40000000, -1073741823, -1], dtype=torch.int32
    )
    raw = qweight.numpy().tobytes()
    buffer = io.BytesIO()
    torch.save(qweight, buffer)

    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        assert archive.read("archive/byteorder") == b"little"
        assert archive.read("archive/version") == b"3\n"
        payload = archive.read("archive/data/0")

    assert payload == raw
    assert payload.hex() == "0000008000000040010000c0ffffffff"
    assert hashlib.sha256(payload).hexdigest() == (
        "bdd7b640dfc023ec83932c35bc9e12fbbecef03f22da9dccf8cc251d88b0f737"
    )
