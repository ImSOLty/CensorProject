from additional import log


def bad_character_table(pattern):
    table = {}
    for i in range(len(pattern) - 1):
        table[pattern[i]] = i
    log(f"BCT: {table}")
    return table


def good_suffix_table(pattern):
    m = len(pattern)
    bpos = [0] * (m + 1)
    shift = [0] * (m + 1)
    # preprocess strong suffix
    i = m
    j = m + 1
    bpos[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = bpos[j]
        i -= 1
        j -= 1
        bpos[i] = j
    # borders
    j = bpos[0]
    for i in range(0, m):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = bpos[j]
    log(f"GST: {shift}")
    return shift


def boyer_moore(word, msg, tables):
    bc_table = tables[0]
    gs_table = tables[1]
    result = []

    i = 0
    while i <= len(msg) - len(word):
        log("|" + msg)
        log("|" + " " * i + word)
        j = len(word) - 1
        while j >= 0 and word[j] == msg[i + j]:
            j -= 1
        if j < 0:
            result.append([word, i])
            log(f"\tFound on {i} position")
            i += gs_table[0]
            log(f"\tGST[0] = {gs_table[0]}")
            log(f"\tShifted by {gs_table[0]}")
        else:
            bad_character = msg[i + j]
            if bad_character in bc_table:
                shift_bc = j - bc_table[bad_character]
            else:
                shift_bc = j + 1
            shift_gs = gs_table[j + 1]
            log(f"\tBCT['{bad_character}'] = {shift_bc}")
            log(f"\tGST[{j + 1}] = {shift_gs}")
            log(f"\t{shift_bc} {'>' if shift_bc > shift_gs else '<'} {shift_gs}")
            i += max(shift_bc, shift_gs)
            log(f"\tShifted by {max(shift_bc, shift_gs)}")

    return result
