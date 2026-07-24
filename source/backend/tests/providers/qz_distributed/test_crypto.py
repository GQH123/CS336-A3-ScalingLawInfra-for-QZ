from scaling_backend.providers.qz_distributed.crypto import encrypt_password


def test_encrypt_password_matches_qz_browser_rsa_compatibility_vector():
    encrypted = encrypt_password("n")

    assert len(encrypted) == 256
    assert encrypted.startswith("0")
    assert encrypted == (
        "09711f4e0dbf11cdd9a2f391feffc66b236727b2e9e24d9b480bafb8fab55986"
        "dab3c0aaa05f404241b96ff8ad44f454f3f0121c4a1399b25039327aac49c4cc"
        "ae653a916b81e8f129d16381c1cc1ea40d0d5e05a75ad2ff8f38de60edd51ac5"
        "cda7449eae6fdce4f1275dfcaed5f66905f368a16151cbf795b404bc7f0c7803"
    )


def test_encrypt_password_leaves_already_encrypted_payload_unchanged():
    already_encrypted = "a" * 256

    assert encrypt_password(already_encrypted) == already_encrypted

