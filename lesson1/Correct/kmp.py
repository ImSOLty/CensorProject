from additional import log


def calculate_lps(pattern):
    lps_result = [0] * len(pattern)
    length = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps_result[i] = length
            i += 1
        else:
            if length != 0:
                length = lps_result[length - 1]
            else:
                lps_result[i] = 0
                i += 1
    log(f"LPS: {lps_result}")
    return lps_result


def KMP(word, msg, lps):
    i = j = 0
    in_msg = []
    shifted = True
    while i < len(msg):
        if shifted:
            log("|" + msg)
            log("|" + " " * (i - j) + word)
            shifted = False

        if msg[i] == word[j]:
            log(f"\t{msg[i]} = {word[j]}")
        else:
            log(f"\t{msg[i]} != {word[j]}")
            shifted = True
        if msg[i] == word[j]:
            i += 1
            j += 1
            if j == len(word):
                in_msg.append([word, i - j])
                log(f"\t\tFound on {i - j} position")
                shifted = True
                log(f"\t\tShifted by {j - lps[j - 1]}")
                j = lps[j - 1]
        else:
            if j != 0:
                log(f"\t\tShifted by {j - lps[j - 1]}")
                j = lps[j - 1]
            else:
                log(f"\t\tShifted by 1")
                i += 1
    return in_msg
