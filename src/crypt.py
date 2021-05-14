# -*- coding: utf-8 -*-
from cryptography.fernet import Fernet


def write_key(key_filename: str):
    """
    Generates a key and save it into a file
    """
    key = Fernet.generate_key()
    with open(key_filename, "wb") as key_file:
        key_file.write(key)


def load_key(key_filename: str):
    return open(key_filename, "rb").read()


def encrypt_file(filename: str, key):
    """
    Given a filename (str) and key (bytes), it encrypts the file and write it
    """
    f = Fernet(key)

    with open(filename, "rb") as file:
        # read all file data
        file_data = file.read()

    # encrypt data
    encrypted_data = f.encrypt(file_data)
    # write the encrypted file
    with open(filename, "wb") as file:
        file.write(encrypted_data)


def decrypt_data(data: bytes, key):
    f = Fernet(key)
    return f.decrypt(data)


def decrypt_file(filename: str, key):
    """
    Given a filename (str) and key (bytes), it decrypts the file and write it
    """
    with open(filename, "rb") as file:
        # read the encrypted data
        encrypted_data = file.read()
    decrypted_data = decrypt_data(encrypted_data, key)
    # write the original file
    with open(filename, "wb") as file:
        file.write(decrypted_data)
