from additional import log

prime = 101
base = 256


def calc_hash(s):
    hash_value = 0
    for char in s:
        hash_value = (hash_value * base + ord(char)) % prime
    return hash_value


def hash_table(words):
    log("Prime: 101\n"
        "Base: 256")
    hwords = {}
    for w in words:
        h = calc_hash(w)
        hwords[h] = w
    log("Words hash values:")
    for k in hwords.keys():
        log(f'\t{k}: {hwords[k]}')
    return hwords


def rabin_karp(words, msg, hwords):
    in_msg = []
    m = len(words[0])

    text_hash = calc_hash(msg[:m])

    for i in range(len(msg) - m + 1):
        log("|" + msg)
        log("|" + " " * i + m * "^")
        log(f"\tSubstring hash: {text_hash}")
        if text_hash in hwords.keys():
            log(f"\tHash equality on {i} position ({hwords[text_hash]})")
            if hwords[text_hash] == msg[i:i + m]:
                in_msg.append([hwords[text_hash], i])
                log(f"\tFound on {i} position ({hwords[text_hash]})")
            else:
                log("\tFalse positive error!")
        if i < len(msg) - m:
            text_hash = ((text_hash - ord(msg[i]) * pow(base, m - 1)) * base + ord(msg[i + m])) % prime
    return in_msg
