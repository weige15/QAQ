import torch


def test_manifest_matches_physical_packed_bytes(checkpoint, manifest):
    qweights = [value for key, value in checkpoint.items() if key.endswith(".qweight")]
    lut4 = [value for key, value in checkpoint.items() if key.endswith(".lut4")]
    lut8 = [value for key, value in checkpoint.items() if key.endswith(".lut8")]
    assert all(value.dtype == torch.int32 for value in qweights)
    qweight_bytes = sum(value.numel() * value.element_size() for value in qweights)
    lut4_bytes = sum(value.numel() * value.element_size() for value in lut4)
    lut8_bytes = sum(value.numel() * value.element_size() for value in lut8)
    record = manifest["artifact"]
    assert qweight_bytes == record["packed_plane_payload_bytes"]
    assert lut4_bytes == record["lookup_bytes"]["4"]
    assert lut8_bytes == record["lookup_bytes"]["8"]
    assert record["selected_packed_plane_bytes"]["4"] == qweight_bytes // 2
    assert record["selected_packed_plane_bytes"]["8"] == qweight_bytes
    assert record["scale_bytes"] == 0
    assert record["total_checkpoint_bytes"] == sum(item["bytes"] for item in record["artifact_file_list"])
