from additional import log


def brute_force(word, msg):
    in_msg = []
    for i in range(len(msg) - len(word) + 1):
        log("|" + msg)
        log("|" + " " * i + word)
        for j in range(len(word)):
            if msg[i + j] == word[j]:
                log(f"\t{msg[i + j]} = {word[j]}")
            else:
                log(f"\t{msg[i + j]} != {word[j]}")

            if msg[i + j] != word[j]:
                break
            elif j == len(word) - 1:
                log(f"\tFound on {i} position")
                in_msg.append([word, i])
    return in_msg
