from __future__ import annotations


def _hex_to_int(hex_string: str) -> int:
    value = hex_string.strip()
    if value.startswith(("0x", "0X")):
        value = value[2:]
    return int(value, 16)


def _int_to_hex(number: int, min_length: int = 0) -> str:
    value = format(number, "x")
    if min_length > 0:
        value = value.zfill(min_length)
    return value


class _BrowserCompatibleRSA:
    def __init__(self, modulus_hex: str, exponent_hex: str):
        self.modulus = _hex_to_int(modulus_hex)
        self.exponent = _hex_to_int(exponent_hex)
        self.chunk_size = 2 * self._bi_high_index(self.modulus)
        self.ciphertext_hex_length = 4 * (self._bi_high_index(self.modulus) + 1)

    @staticmethod
    def _bi_high_index(number: int) -> int:
        if number == 0:
            return 0
        return (number.bit_length() + 15) // 16 - 1

    @staticmethod
    def _encode_block(byte_array: list[int], start: int, chunk_size: int) -> int:
        block = 0
        digit_index = 0
        for idx in range(start, start + chunk_size, 2):
            byte1 = byte_array[idx] if idx < len(byte_array) else 0
            byte2 = byte_array[idx + 1] if idx + 1 < len(byte_array) else 0
            digit = byte1 + (byte2 << 8)
            block += digit << (16 * digit_index)
            digit_index += 1
        return block

    def encrypt_string(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        byte_array = [ord(char) for char in plaintext]
        while len(byte_array) % self.chunk_size != 0:
            byte_array.append(0)

        result_parts = []
        for start in range(0, len(byte_array), self.chunk_size):
            block = self._encode_block(byte_array, start, self.chunk_size)
            encrypted = pow(block, self.exponent, self.modulus)
            result_parts.append(_int_to_hex(encrypted, self.ciphertext_hex_length))
        return " ".join(result_parts)


class PasswordEncryptor:
    EXPONENT = "010001"
    MODULUS = (
        "008aed7e057fe8f14c73550b0e6467b023616ddc8fa91846d2613cdb7f7621e3"
        "cada4cd5d812d627af6b87727ade4e26d26208b7326815941492b2204c3167ab"
        "2d53df1e3a2c9153bdb7c8c2e968df97a5e7e01cc410f92c4c2c2fba529b"
        "3ee988ebc1fca99ff5119e036d732c368acf8beba01aa2fdafa45b21e4de49"
        "28d0d403"
    )

    def __init__(self):
        self._rsa = _BrowserCompatibleRSA(self.MODULUS, self.EXPONENT)

    def encrypt(self, password: str) -> str:
        if self.is_encrypted(password):
            return password
        return self._rsa.encrypt_string(password).replace(" ", "")

    @staticmethod
    def is_encrypted(password: str) -> bool:
        return (
            254 <= len(password) <= 256
            and all(char in "0123456789abcdefABCDEF" for char in password)
        )


def encrypt_password(password: str) -> str:
    return PasswordEncryptor().encrypt(password)

