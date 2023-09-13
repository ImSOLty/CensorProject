import re

from hstest import *
from enum import Enum
from typing import List


class Alg(str, Enum):
    BRUTE_FORCE = "brute-force"
    KMP = "knuth-morris-pratt"
    BOYER_MOORE = "boyer-moore"
    RABIN_KARP = "rabin-karp"
    AHO_CORASICK = "aho-corasick"
    AHO_CORASICK_WILDCARD = "aho-corasick-wildcard"


naming = {
    Alg.BRUTE_FORCE: "Brute Force",
    Alg.KMP: "Knuth-Morris-Pratt",
    Alg.BOYER_MOORE: "Boyer-Moore",
    Alg.RABIN_KARP: "Rabin-Karp",
    Alg.AHO_CORASICK: "Aho-Corasick",
    Alg.AHO_CORASICK_WILDCARD: "Aho-Corasick with Wildcards",
}

GOOD_FEEDBACK = "No inappropriate content detected"


def simple_test_case(i: str, p: str, o: str, a: Alg, cause):
    files = {'test.txt': i}
    if ',' in p:
        files['words.txt'] = p.replace(',', '\n')
        args = ["-f", "test.txt", "-w", "words.txt", "-a", a.value, '-m']
    else:
        args = ["-f", "test.txt", "-w", p, "-a", a.value]
    return [TestCase(files=files,
                     args=args,
                     attach=(o, i, p, a.value, cause, False),
                     check_function=simple_check),
            TestCase(files=files,
                     args=args + ['-l'],
                     attach=(o, i, p, a.value, cause, True),
                     check_function=simple_check)]


def format_test_case(i: str, p: str, a: Alg, ok: bool):
    files = {'test.txt': i}
    if ',' in p:
        files['words.txt'] = p.replace(',', '\n')
        args = ["-f", "test.txt", "-w", "words.txt", "-a", a.value, '-m', '-l']
    else:
        args = ["-f", "test.txt", "-w", p, "-a", a.value, "-l"]
    return TestCase(files=files,
                    args=args,
                    attach=(a, ok),
                    check_function=log_format_check)


def processing_test_case(i: str, p: str, a: Alg, reveal: str):
    funcs = {
        Alg.BRUTE_FORCE: bf_order,
        Alg.KMP: kmp_order,
        Alg.BOYER_MOORE: bm_order,
        Alg.RABIN_KARP: rk_order,
    }
    if a == Alg.RABIN_KARP:
        order = funcs[a](i.split('\n'), p.split(','))
    elif a == Alg.AHO_CORASICK:
        order = ac_order(i.split('\n'), p.split(','), False)
    elif a == Alg.AHO_CORASICK_WILDCARD:
        order = ac_order(i.split('\n'), [p], True)
    else:
        order = funcs[a](i.split('\n'), p)

    files = {'test.txt': i}
    if ',' in p:
        files['words.txt'] = p.replace(',', '\n')
        args = ["-f", "test.txt", "-w", "words.txt", "-a", a.value, '-l', '-m']
    else:
        args = ["-f", "test.txt", "-w", p, "-a", a.value, '-l']
    return TestCase(files=files,
                    args=args,
                    attach=(i, order, a, p, reveal),
                    check_function=processing_check)


def simple_check(reply: str, attach) -> CheckResult:
    orig = reply.strip()
    reply = reply.strip().lower()
    output, inp, pattern, alg, cause, logging = attach
    if logging:
        reply = re.split(r"===+", reply)[-1].strip()
        orig = re.split(r"===+", orig)[-1].strip()
        if output == "":
            output = GOOD_FEEDBACK
    if output.lower() != reply:
        if cause == "":
            return CheckResult.wrong(f"Incorrect result.\n"
                                     f"Algorithm: {naming[alg]}\n"
                                     f"Pattern{'s' if ',' in pattern else ''}: {pattern}\n"
                                     f"\nText:\n{inp}\n"
                                     f"\nExpected:\n{output}\n"
                                     f"\nGot:\n{orig}")
        else:
            return CheckResult.wrong(f"Incorrect result.\n"
                                     f"Algorithm: {naming[alg]}\n"
                                     f"Pattern{'s' if ',' in pattern else ''}: {pattern}\n"
                                     f"Possible cause: {cause}")
    return CheckResult.correct()


def log_format_check(reply: str, attach) -> CheckResult:
    reply = reply.strip().lower()
    alg, ok = attach
    seps = [part.strip() for part in re.split(r"===+", reply)]

    parts = 4

    if len(seps) != parts:
        return CheckResult.wrong(
            f"(Algorithm: {naming[alg]}) If logging is enabled there should be {parts} separated "
            f"parts in output : Algorithm, Preprocessing, Processing, Result. Found: {len(seps)}")
    if ok and seps[-1] != GOOD_FEEDBACK.lower():
        return CheckResult.wrong(
            f"If there is no inappropriate content in text when logging is enabled "
            f"feedback should be \"{GOOD_FEEDBACK}\"")
    if alg.value not in seps[0] and naming[alg].lower() not in seps[0]:
        return CheckResult.wrong(f"First separated part in output should contain algorithm name or argument")
    return CheckResult.correct()


def processing_check(reply: str, attach) -> CheckResult:
    inp, order, algorithm, pattern, reveal = attach

    funcs = {
        Alg.KMP: preprocessing_kmp_check,
        Alg.BOYER_MOORE: preprocessing_bm_check,
        Alg.RABIN_KARP: preprocessing_rk_check,
        Alg.AHO_CORASICK: preprocessing_ac_check,
        Alg.AHO_CORASICK_WILDCARD: preprocessing_ac_check,
    }

    orig_text = reply.strip()
    prepr, pr = re.split(r"===+", orig_text)[1:3]
    lines_reply = re.findall(r"[Ll]ine: \"(.*)\"\n", pr)
    lines_actual = inp.split('\n')
    if lines_reply != lines_actual:
        if reveal == '':
            return CheckResult.wrong(f'Incorrect (Line: \"*line*\") substrings found in processing part.\n'
                                     f'Expected: {["Line: " + l for l in lines_actual]}\n'
                                     f'Got: {["Line: " + l for l in lines_reply]}')
        else:
            return CheckResult.wrong(
                f'Incorrect (Line: \"*line*\") substrings found in processing part. Case: {reveal}\n')
    pr = [line.strip() for line in re.split(r"[Ll]ine: \".*\"\n", "a" + pr)[1:]]

    for i in range(len(lines_actual)):
        if not re.match(r'(\|.*\n\|.*\n.*)*', pr[i]):
            return CheckResult.wrong(f"Incorrect log formatting found for \"{lines_actual[i]}\" line. "
                                     f"Please, follow the instructions from the description.")
    if algorithm == Alg.BRUTE_FORCE:
        return processing_alg_check(pr, order, reveal, lines_actual, algorithm, pattern)
    if algorithm == Alg.AHO_CORASICK:
        result = funcs[algorithm](prepr, pattern, reveal, False)
    elif algorithm == Alg.AHO_CORASICK_WILDCARD:
        result = funcs[algorithm](prepr, pattern, reveal, True)
    else:
        result = funcs[algorithm](prepr, pattern, reveal)
    if not result.is_correct or len(order) == 0:
        return result
    return processing_alg_check(pr, order, reveal, lines_actual, algorithm, pattern)


def processing_alg_check(reply: List, order: List, reveal: str, lines_actual: List, alg, pattern) -> CheckResult:
    # C-ARRAYS FOR LINES IN AHO_CORASICK_WILDCARD
    if alg == Alg.AHO_CORASICK_WILDCARD:
        c_arrays = [line[-1] for line in order]
        order = [without[:-1] for without in order]

    mod_order = [[s.replace('\t', '').lower() for s in inner_list] for inner_list in order]
    # for each line
    for i in range(len(order)):
        exp = []
        got = []
        line_parts_out = re.findall('\|.*\n\|.*\n', reply[i])
        line_parts_in = re.split('\|.*\n\|.*\n', reply[i])[1:]
        # for each part in exact order
        for j in range(len(order[i]) - 2):
            if j % 2 == 0:
                check_str = line_parts_out[int(j / 2)]
            else:
                check_str = line_parts_in[int((j - 1) / 2)]
            exp.append(order[i][j])
            got.append(check_str)
            for stm in mod_order[i][j].split('\n'):
                if stm not in check_str.replace('\t', '').strip().lower():
                    exp.append("...")
                    got.append("...")
                    exp = "\n".join(exp).replace('\n\n', '\n')
                    got = "".join(got)
                    if reveal == '':
                        return CheckResult.wrong(
                            f"Incorrect log content/formatting found for \"{lines_actual[i]}\" line. "
                            f"Pattern: {pattern}\n"
                            f"Algorithm: {naming[alg]}\n"
                            f"\nExample of expected:\n{exp}"
                            f"\nGot: \n{got.strip()}")
                    else:
                        return CheckResult.wrong(f"Incorrect log content/formatting found. "
                                                 f"Algorithm: {naming[alg]}. Case: {reveal}")

        # CHECK C-ARRAYS FOR LINES IN AHO_CORASICK_WILDCARD
        if alg == Alg.AHO_CORASICK_WILDCARD and c_arrays[i] not in "".join(reply[i]).lower():
            if reveal == '':
                return CheckResult.wrong(
                    f"Incorrect c-array found for \"{lines_actual[i]}\" line and pattern: {pattern}.\n"
                    f"Algorithm: {naming[alg]}\n"
                    f"\nExample of expected:\n{c_arrays[i]}")
            else:
                return CheckResult.wrong(f"Incorrect c-array found. Algorithm: {naming[alg]}. Case: {reveal}")
    return CheckResult.correct()


prime = 101
base = 256


def calc_hash(s):
    hash_value = 0
    for char in s:
        hash_value = (hash_value * base + ord(char)) % prime
    return hash_value


def bf_order(lines_actual: List, word: str) -> List:
    order = []
    for line in lines_actual:
        for_line = []
        for i in range(len(line) - len(word) + 1):
            for_line.append("|" + line + "\n|" + " " * i + word)
            inside = ""
            for j in range(len(word)):
                if line[i + j] == word[j]:
                    inside += f"\n\t{line[i + j]} = {word[j]}"
                else:
                    inside += f"\n\t{line[i + j]} != {word[j]}"

                if line[i + j] != word[j]:
                    break
                elif j == len(word) - 1:
                    inside += f"\n\tFound on {i} position"
            for_line.append(inside)
        order.append(for_line)
    return order


def lps_function(pattern: str):
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
    return lps_result


def preprocessing_kmp_check(reply: str, pattern: str, reveal: str) -> CheckResult:
    p = f"LPS: {lps_function(pattern)}"
    if p.lower() not in reply.lower():
        if reveal == '':
            return CheckResult.wrong(f"Incorrect preprocessing content/formatting found for \"{pattern}\" pattern. "
                                     f"Algorithm: {naming[Alg.KMP]}\n"
                                     f"\nExpected \"{p}\" in output.\n"
                                     f"Got:\n"
                                     f"{reply.strip()}")
        else:
            return CheckResult.wrong(f"Incorrect preprocessing content/formatting found for patterns. "
                                     f"Algorithm: {naming[Alg.KMP]}. Case: {reveal}")
    return CheckResult.correct()


def kmp_order(lines_actual: List, word: str) -> List:
    order = []
    lps = lps_function(word)
    for line in lines_actual:
        i = j = 0
        for_line = []
        shifted = True
        while i < len(line) - len(word) + 1:
            if shifted:
                for_line.append("|" + line + "\n|" + " " * (i - j) + word)
                shifted = False
            inside = ""
            if line[i] == word[j]:
                inside += f"\n\t{line[i]} = {word[j]}"
            else:
                inside += f"\n\t{line[i]} != {word[j]}"
                shifted = True
            if line[i] == word[j]:
                i += 1
                j += 1
                if j == len(word):
                    inside += f"\n\tFound on {i - j} position"
                    shifted = True
                    inside += f"\n\tShifted by {j - lps[j - 1]}"
                    for_line.append(inside)
                    j = lps[j - 1]
            else:
                if j != 0:
                    inside += f"\n\tShifted by {j - lps[j - 1]}"
                    for_line.append(inside)
                    j = lps[j - 1]
                else:
                    inside += f"\n\tShifted by 1"
                    for_line.append(inside)
                    i += 1
        order.append(for_line)
    return order


def bad_character_table(pattern):
    table = {}
    for i in range(len(pattern) - 1):
        table[pattern[i]] = i
    return table


def good_suffix_table(pattern):
    m = len(pattern)
    bpos = [0] * (m + 1)
    shift = [0] * (m + 1)
    i, j = m, m + 1
    bpos[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = bpos[j]
        i -= 1
        j -= 1
        bpos[i] = j
    j = bpos[0]
    for i in range(0, m):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = bpos[j]
    return shift


def bm_order(lines_actual: List, word: str):
    bc_table, gs_table = bad_character_table(word), good_suffix_table(word)
    order = []
    for line in lines_actual:
        for_line = []
        i = 0
        while i <= len(line) - len(word):
            for_line.append(f"|{line}\n|{' ' * i}{word}")
            inside = ''
            j = len(word) - 1
            while j >= 0 and word[j] == line[i + j]:
                j -= 1
            if j < 0:
                inside += f"\tFound on {i} position"
                i += gs_table[0]
                inside += f"\n\tGST[0] = {gs_table[0]}"
                inside += f"\n\tShifted by {gs_table[0]}"
            else:
                bad_character = line[i + j]
                shift_bc = j - bc_table[bad_character] if bad_character in bc_table else j + 1
                shift_gs = gs_table[j + 1]
                inside += f"\tBCT['{bad_character}'] = {shift_bc}"
                inside += f"\n\tGST[{j + 1}] = {shift_gs}"
                i += max(shift_bc, shift_gs)
                inside += f"\n\tShifted by {max(shift_bc, shift_gs)}"
            for_line.append(inside)
        order.append(for_line)
    return order


def preprocessing_bm_check(reply: str, pattern: str, reveal: str) -> CheckResult:
    for p in [f"BCT: {bad_character_table(pattern)}", f"GST: {good_suffix_table(pattern)}"]:
        if p.lower() not in reply.lower():
            if reveal == '':
                return CheckResult.wrong(f"Incorrect preprocessing content/formatting found for patterns: {pattern}"
                                         f"Algorithm: {naming[Alg.BOYER_MOORE]}\n"
                                         f"\nExpected \"{p}\" in output.\n"
                                         f"Got:\n"
                                         f"{reply.strip()}")
            else:
                return CheckResult.wrong(f"Incorrect preprocessing content/formatting found for patterns. "
                                         f"Algorithm: {naming[Alg.BOYER_MOORE]}. Case: {reveal}")
    return CheckResult.correct()


def rk_order(lines_actual: List, words: List) -> List:
    hwords = {}
    for w in words:
        h = calc_hash(w)
        hwords[h] = w

    order = []

    for line in lines_actual:
        for_line = []
        m = len(words[0])
        text_hash = calc_hash(line[:m])

        for i in range(len(line) - m + 1):
            for_line.append("|" + line + "\n|" + " " * i + m * "^")
            inside = ""
            inside += f"\tHash: {text_hash}"
            if text_hash in hwords.keys():
                inside += f"\n\tHash equality on {i} position ({hwords[text_hash]})"
                if hwords[text_hash] == line[i:i + m]:
                    inside += f"\n\tFound on {i} position ({hwords[text_hash]})"
                else:
                    inside += "\n\tFalse positive error!"
            for_line.append(inside)
            if i < len(line) - m:
                text_hash = (text_hash - ord(line[i]) * pow(base, m - 1, prime)) % prime
                text_hash = (text_hash * base + ord(line[i + m])) % prime
        order.append(for_line)
    return order


def preprocessing_rk_check(reply: str, pattern: str, reveal: str) -> CheckResult:
    if str(prime) not in reply.lower() or str(base) not in reply.lower():
        if reveal == '':
            return CheckResult.wrong(
                f"Incorrect preprocessing content/formatting found. "
                f"Expected prime({prime}) and base({base}) numbers in output."
                f"Algorithm: {naming[Alg.RABIN_KARP]}\n"
                f"Got:\n"
                f"{reply.strip()}")
        else:
            return CheckResult.wrong(
                f"Incorrect preprocessing content/formatting found. "
                f"Expected prime({prime}) and base({base}) numbers in output."
                f"Algorithm: {naming[Alg.RABIN_KARP]}\n")
    for p in pattern.lower().split(','):
        if f"{calc_hash(p)}: {p}" not in reply.lower():
            if reveal == '':
                return CheckResult.wrong(f"Incorrect preprocessing content/formatting found for patterns: {pattern}. "
                                         f"Algorithm: {naming[Alg.RABIN_KARP]}\n"
                                         f"\nExpected \"{p}\" in output.\n"
                                         f"Got:\n"
                                         f"{reply.strip()}")
            else:
                return CheckResult.wrong(f"Incorrect preprocessing content/formatting found for patterns. "
                                         f"Algorithm: {naming[Alg.RABIN_KARP]}. Case: {reveal}")
    return CheckResult.correct()


global id_out
id_out = 0


class AhoNode:
    def __init__(self, lvl, s):
        global id_out
        self.lvl = lvl
        self.id = id_out
        self.s = s
        self.goto = {}
        self.out = []
        self.fail = None

    def print(self):
        s = ''
        for i in self.goto.keys():
            s += "_" * 4 * self.lvl
            if self.goto[i] is not None:
                s += f"{i, self.goto[i].id, self.goto[i].fail.id}\n"
                s += self.goto[i].print()
        return s


def aho_create_forest(patterns, indices):
    root = AhoNode(0, '')
    for path in range(len(patterns)):
        lvl = 0
        node = root
        for symbol in patterns[path]:
            lvl += 1
            if symbol not in node.goto.keys():
                global id_out
                id_out += 1
            node = node.goto.setdefault(symbol, AhoNode(lvl, node.s + symbol))
        if len(indices) > 0:
            node.out.append([patterns[path], indices[path]])
        else:
            node.out.append([patterns[path]])
    return root


def aho_create_statemachine(patterns, indices=None):
    if indices is None:
        indices = []

    global id_out
    id_out = 0

    root = aho_create_forest(patterns, indices)
    queue = []
    for node in root.goto.values():
        queue.append(node)
        node.fail = root
    while len(queue) > 0:
        rnode = queue.pop(0)
        for key, unode in rnode.goto.items():
            queue.append(unode)
            fnode = rnode.fail
            while fnode is not None and key not in fnode.goto:
                fnode = fnode.fail
            unode.fail = fnode.goto[key] if fnode else root
            unode.out += unode.fail.out
    return root


def ac_order(lines_actual: List, words: List, joker=False):
    order = []
    if joker:
        matches = [(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'[^\*]+', words[0])]
        word, indices = zip(*matches)
        root = aho_create_statemachine(list(word), [i[0] for i in indices])
    else:
        root = aho_create_statemachine(words)

    for line in lines_actual:
        for_line = [f"|{line}\n|"]
        c = [0] * len(line)
        node = root
        inside = ''
        for i in range(len(line)):
            while node is not None and line[i] not in node.goto:
                trans = f"\n\tTransition: \'{node.s}\'->\'{node.fail.s if node.fail is not None else ''}\'"
                if "''->''" not in trans:
                    inside += trans
                node = node.fail
                if node is not None and node.fail is not None:
                    for_line.append(inside)
                    for_line.append(f"|{line}\n|{' ' * (i - node.lvl) + '^' * node.lvl}")
                    inside = ''
            if node is None:
                node = root
                continue
            trans = (f"\n\tTransition: \'{node.s if node is not None else ''}\'->"
                     f"\'{node.goto[line[i]].s if node.goto[line[i]] is not None else ''}\'")
            if "''->''" not in trans:
                inside += trans
            node = node.goto[line[i]]
            for_line.append(inside)
            for_line.append(f"|{line}\n|{' ' * (i - node.lvl + 1) + '^' * node.lvl}")
            inside = ''
            for pattern in node.out:
                if joker and i - len(pattern[0]) - pattern[1] + 1 >= 0:
                    c[i - len(pattern[0]) - pattern[1] + 1] += 1
                inside += f"\n\tFound on {i - len(pattern[0]) + 1} position ({pattern[0]})"
        if joker:
            for_line.append(str(c))
        order.append(for_line)
    return order


def preprocessing_ac_check(reply: str, pattern: str, reveal: str, joker=False) -> CheckResult:
    if joker:
        matches = [(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'[^\*]+', pattern)]
        word, indices = zip(*matches)
        ac_wild = ""
        for i in range(len(word)):
            ac_wild += f'\t{word[i]}({indices[i][0]})\n'
        for part in ac_wild.split('\n'):
            if part not in reply:
                if reveal == '':
                    raise WrongAnswer(f"Incorrect part-indices content/formatting found for pattern: {pattern}. "
                                      f"Algorithm: {naming[Alg.AHO_CORASICK_WILDCARD]}\n"
                                      f"\nExpected \"{ac_wild.strip()}\" in output.\n"
                                      f"Got:\n"
                                      f"{reply.strip()}")
                else:
                    raise WrongAnswer(f"Incorrect part-indices content/formatting found for pattern. "
                                      f"Algorithm: {naming[Alg.AHO_CORASICK_WILDCARD]}. Case: {reveal}")

    state_machine = aho_create_statemachine(list(word), [i[0] for i in indices]) if joker else aho_create_statemachine(
        pattern.lower().split(','))
    trie_orig = state_machine.print().lower().strip()
    trie = trie_orig.replace('_', '')
    for node in trie.strip().split('\n'):
        if node not in reply.lower():
            if reveal == '':
                return CheckResult.wrong(f"Incorrect trie content/formatting found for patterns: {pattern}. "
                                         f"Algorithm: {naming[Alg.AHO_CORASICK_WILDCARD if joker else Alg.AHO_CORASICK]}\n"
                                         f"\nExpected:\n{trie_orig}\n"
                                         f"Got:\n"
                                         f"{reply.strip()}")
            else:
                return CheckResult.wrong(f"Incorrect preprocessing content/formatting found for patterns. "
                                         f"Algorithm: {naming[Alg.AHO_CORASICK_WILDCARD if joker else Alg.AHO_CORASICK]}. Case: {reveal}")
    return CheckResult.correct()


class TestAlgorithms(StageTest):

    def __init__(self, stages):
        super().__init__()
        self.stages = stages

    def generate(self):
        tests = []
        if 1 in self.stages:
            tests += format1(Alg.BRUTE_FORCE) + generate_testcases(stage1_tests, Alg.BRUTE_FORCE) + \
                     general(Alg.BRUTE_FORCE) + generate_testcases(stage1_specific, Alg.BRUTE_FORCE)
        if 2 in self.stages:
            tests += format1(Alg.KMP) + generate_testcases(stage2_tests, Alg.KMP) + \
                     general(Alg.KMP) + generate_testcases(stage2_specific, Alg.KMP)
        if 3 in self.stages:
            tests += format1(Alg.BOYER_MOORE) + generate_testcases(stage3_tests, Alg.BOYER_MOORE) + \
                     general(Alg.BOYER_MOORE) + generate_testcases(stage3_specific, Alg.BOYER_MOORE)
        if 4 in self.stages:
            tests += format1(Alg.RABIN_KARP) + format2(Alg.RABIN_KARP) + \
                     generate_testcases(stage4_tests, Alg.RABIN_KARP) + \
                     generate_testcases(stage4_specific, Alg.RABIN_KARP)
        if 5 in self.stages:
            tests += format1(Alg.AHO_CORASICK) + format2(Alg.AHO_CORASICK) + \
                     generate_testcases(stage5_tests, Alg.AHO_CORASICK) + \
                     generate_testcases(stage5_specific, Alg.AHO_CORASICK)
        if 6 in self.stages:
            tests += format1(Alg.AHO_CORASICK_WILDCARD) + generate_testcases(stage6_tests, Alg.AHO_CORASICK_WILDCARD)
        return tests


if __name__ == '__main__':
    TestAlgorithms([5]).run_tests()


def generate_testcases(cases: List, alg: Alg):
    testcases = []
    for case in cases:
        if len(case) > 4 and case[4]:
            testcases.extend(simple_test_case(case[0], case[1], case[2], alg, case[3]))
            testcases.append(processing_test_case(case[0], case[1], alg, case[3]))
        elif case[2] == '':
            testcases.append(processing_test_case(case[0], case[1], alg, case[3]))
        else:
            testcases.extend(simple_test_case(case[0], case[1], case[2], alg, case[3]))
    return testcases


# ['input','pattern', (empty if logging checked, else output),'(empty if revealed, else case description)']
stage1_tests = [
    # SIMPLE TESTS
    ['test', 'e', '', ''],
    ['hello', 'o', '', ''],
    # TESTS WITH MULTIPLE STRINGS
    ['foo\nbar', 'b', '', ''],
    ['he\nty', 'e', '', 'Multiple messages in text.'],
    # LONGER INAPPROPRIATE
    ['wrrld', 'rl', '', ''],
    ['dddda', 'dda', '', 'Inappropriate word is longer than 1 symbol.'],
    # MULTIPLE CASES
    ['banana', 'a', '', ''],
    ['sdsdf', 's', '', 'Inappropriate word enters message multiple times.'],
]

stage1_specific = [
    ['bruteforcespecific', 'force', 'brute*****specific', 'Hidden case'],
    ['heyheyahey', 'hey', '******a***', 'Hidden case'],
    ['abcdcbcd', 'cd', 'ab**cb**', 'Hidden case'],
    ['cbabcaacb', 'a', 'cb*bc**cb', 'Hidden case'],
    ['abcbabcd\nabcdabc', 'bcd', 'abcba***\na***abc', 'Hidden case'],
    ['abc\n' * 4 + 'abc', 'abc', '***\n' * 4 + '***', 'Hidden case'],
]

stage2_tests = [
    # SIMPLE TESTS
    ['test', 'e', '', ''],
    ['heyllo', 'y', '', ''],
    # TESTS WITH MULTIPLE STRINGS
    ['foo\nbar', 'b', '', ''],
    ['he\nty', 'e', '', 'Multiple messages in text.'],
    # LONGER INAPPROPRIATE
    ['wrrld', 'rl', '', ''],
    ['ddadd', 'da', '', 'Inappropriate word is longer than 1 symbol.'],
    # MULTIPLE CASES
    ['banana', 'a', '', ''],
    ['sdsdf', 's', '', 'Inappropriate word enters message multiple times.'],
    # INAPPROPRIATE WORD HAS SAME PREFIXES AND SUFFIXES
    ['abcabda', 'abda', '', ''],
    ['bcbccbcbdc', 'cbdc', '', 'Inappropriate word has same prefixes and suffixes.'],
    # PREFIX AND SUFFIX ARE LONG
    ['abcdeabcdadbabcdeabcd', 'abcdeabcd', '', ''],
    ['adbabcdefgabcdefgadbabcdefgabcdefgadb', 'abcdefgabcdefg', '', 'Prefix and suffix are long.'],

    # ONLY LPS
    # PATTERN IS A PALINDROME WITH LENGTH > 1
    ['aabcbaa', 'aabcbaa', '', ''],
    ['aaaccccaaa', 'aaaccccaaa', '', 'Pattern is a palindrome with length>1.'],
    # PATTERN HAS NO REPEATED LETTERS
    ['abcd', 'abcd', '', ''],
    ['qwertyuiop', 'qwertyuiop', '', 'Pattern has no repeated letters.'],
    # MULTIPLE SAME PREFIXES AND SUFFIXES
    ['cbcbcc', 'cbcbcc', '', ''],
    ['aabaabaaa', 'aabaabaaa', '', 'Multiple prefixes and suffixes.'],
]

stage2_specific = [
    # KMP SPECIFIC
    ['ababababababababababababab', 'ababababab', '*' * 26, 'Hidden case'],
    ['ababababab', 'ababababab', '', 'Hidden case'],
    ['aaaaaaabaaaaaaabaaaaaaab', 'aaaab', 'aaa*****' * 3, 'Hidden case'],
    ['aaaab', 'aaaab', '', 'Hidden case'],
    ['acccccbccdcbccbcc', 'ccbcc', 'accc*****dcb*****', 'Hidden case'],
    ['ccbcc', 'ccbcc', '', 'Hidden case'],
    ['abcdabcdabcde', 'abcdabcde', 'abcd*********', 'Hidden case'],
    ['abcdabcde', 'abcdabcde', '', 'Hidden case'],
    ['abbacabbbabcdab\nbaabbadab', 'abba', '****cabbbabcdab\nba****dab', 'Hidden case'],
    ['abba', 'abba', '', 'Hidden case'],
    ['abcdefabcdefabcdefabcdef\n' * 15 + 'a', 'abcdefabcdef', '************************\n' * 15 + 'a', 'Hidden case'],
    ['abcdeabcde', 'abcdeabcde', '', 'Hidden case'],
]

stage3_tests = [
    # SIMPLE TESTS
    ['test', 'e', '', ''],
    ['heyllo', 'y', '', ''],
    # TESTS WITH MULTIPLE STRINGS
    ['foo\nbar', 'b', '', ''],
    ['he\nty', 'e', '', 'Multiple messages in text.'],
    # LONGER INAPPROPRIATE
    ['wrrld', 'rl', '', ''],
    ['ddadd', 'da', '', 'Inappropriate word is longer than 1 symbol.'],
    # MULTIPLE CASES
    ['banana', 'a', '', ''],
    ['sdsdf', 's', '', 'Inappropriate word enters message multiple times.'],

    # BCT beneficial
    # The alphabet in the text and pattern is relatively small.
    ['abaababaa', 'aa', '', ''],
    ['ddedddede', 'dd', '', 'The alphabet in the text and pattern is relatively small.'],
    # The pattern is long and contains fewer repetitions of characters.
    ['bcdefgabcdefg', 'bcdefg', '', ''],
    ['qwertyuioqwertyuiop', 'qwertyuiop', '', 'The pattern is long and contains fewer repetitions of characters.'],
    # The text is much larger than the pattern.
    ['bcddcbbcdbdbcdbaaa', 'aaa', '', ''],
    ['abcdefgaqwabcdefgaqw', 'qw', '', 'The text is much larger than the pattern.'],

    # GST beneficial
    # Many repeating suffixes and/or prefixes in the search text.
    ['banaanabana', 'ana', '', ''],
    ['abracadabracadabracadabra', 'abra', '', 'Many repeating suffixes and/or prefixes in the search text.'],
    # Pattern contains multiple characters that repeat within the pattern.
    ['mississipp', 'iss', '', ''],
    ['babababababababa', 'baba', '', 'Pattern contains multiple characters that repeat within the pattern.'],

    # ONLY BCT and GST
    ['abracadabra', 'abracadabra', '', ''],
    ['xyxxyzxyzxyx', 'xyxxyzxyzxyx', '', ''],
    ['banana', 'banana', '', ''],
    ['abcabcabc', 'abcabcabc', '', ''],
    ['aaaaaaaaaa', 'aaaaaaaaaa', '', ''],
]

stage3_specific = [
    # BOYER-MOORE SPECIFIC
    ['abracadabra ' * 9 + 'abracadabra', 'abracadabra', '*********** ' * 9 + '***********', 'Hidden case'],
    ['abracadabra', 'abracadabra', '', 'Hidden case'],
    ['xyxxyzxyzxyxxyxxyzxyzxyxxyxxyzxyzxyxxyxxyzxyzxyx', 'xyxxyzxyzxyx', '*' * 48, 'Hidden case'],
    ['xyxxyzxyzxyx', 'xyxxyzxyzxyx', '', 'Hidden case'],
    ['banana apple ' * 9 + 'banana apple', 'banana', '****** apple ' * 9 + '****** apple', 'Hidden case'],
    ['mississippi ' * 9 + 'mississippi', 'mississippi', '*********** ' * 9 + '***********', 'Hidden case'],
    ['mississippi', 'mississippi', '', 'Hidden case'],
    ['aabaabaaaabaabaaabaabaaaabaabaaabaabaaaabaabaaabaabaaaabaabaaabaabaaa', 'aabaabaaa', '*' * 69, 'Hidden case'],
    ['aabaabaaa', 'aabaabaaa', '', 'Hidden case'],
    ['abracadabra ' * 9 + 'abracadabra', 'cadabra', 'abra******* ' * 9 + 'abra*******', 'Hidden case'],
    ['cadabra', 'cadabra', '', 'Hidden case'],
    ['xyxxyzxyzxyxxyxxyzxyzxyxxyxxyzxyzxyxxyxxyzxyzxyx', 'xyzxyx', 'xyxxyz******' * 4, 'Hidden case'],
    ['xyzxyx', 'xyzxyx', '', 'Hidden case'],
    ['mississippi ' * 9 + 'mississippi', 'sip', 'missis***pi ' * 9 + 'missis***pi', 'Hidden case'],
]

stage4_tests = [
    # SIMPLE TESTS
    ['test', 'e', '', ''],
    ['heyllo', 'y', '', ''],
    # TESTS WITH MULTIPLE STRINGS
    ['foo\nbar', 'b', '', ''],
    ['he\nty', 'e', '', 'Multiple messages in text.'],
    # LONGER INAPPROPRIATE
    ['wrrld', 'rl', '', ''],
    ['ddadd', 'da', '', 'Inappropriate word is longer than 1 symbol.'],
    # MULTIPLE CASES
    ['banana', 'a', '', ''],
    ['sdsdf', 's', '', 'Inappropriate word enters message multiple times.'],
    # REPETITIVE PATTERNS
    ['abbcdeabbcdeabbcde', 'abb,cde', '', ''],
    ['ababadababad', 'ab,ad', '', 'Text contains repetitive patterns.'],
    # NUMBER OF PATTERNS > 2
    ['abbbdcdbeeff', 'ab,cd,ef', '', ''],
    ['abacdcabaefecdc', 'aba,cdc,efe', '', 'Number of patterns is greater than 2.'],
    # OVERLAPPING PATTERNS
    ['abcdedcdbdcabcd', 'ab,bc,cd', '', ''],
    ['aabbccaabbccaaa', 'aabb,bbcc,ccaa', '', 'Some patterns are overlapping.'],
    # PATTERNS HAVE SAME CHARACTERS BUT DIFFERENT ORDERS
    ['baabbabab', 'ab,ba', '', ''],
    ['abcabcabcabcabc', 'abc,cab,bca', '', 'Patterns have same characters but different orders.'],
    # PATTERNS ARE LONG
    ['abcdefghijklmnopqrst', 'abcdefgh,mnopqrst', '', ''],
    ['aaaaaaaaabbbbbbbbbccccccccc', 'aaaaaaaaa,bbbbbbbbb,ccccccccc', '', 'Patterns are long.'],
    # HASH COLLISION
    ['abaoaaba', 'aab,aaa', 'abao***a', ''],
    ['abaoaaba', 'aab,aaa', '', ''],
    ['dimdkfdtyemlaabbao', 'aju,laa', 'dimdkfdtyem***bbao', 'Possible hash collision.'],
    ['dimdkfdtyemlaabbao', 'aju,laa', '', 'Possible hash collision.'],
]

stage4_specific = [
    # RABIN-KARP SPECIFIC
    ['the cat in the hat wears a hat, while it is a cat', 'hat,cat,the',
     '*** *** in *** *** wears a ***, while it is a ***', 'Hidden case'],
    ['xyhjklvqet', 'hjkl,vqet', 'xy********', 'Hidden case'],
    ['level', 'eve,ele', 'l***l', 'Hidden case'],
    ['abcdeabcde', 'deabc,cdeab', 'ab******de', 'Hidden case'],
    ['aaaaaaa', 'aaa,aaa', '*******', 'Hidden case'],
    ['xzyzxyyxyyzxzyxyxyzxyyzxzxyzxyyxyzxyzzyxyzyxyzyx', 'xyz,xzy,yxz,yzx,zxy,zyx',
     '******yxy******y**************y*****************', 'Hidden case'],
]

stage5_tests = [
    # SIMPLE TESTS
    ['test', 'e', '', ''],
    ['heyllo', 'y', '', ''],
    # TESTS WITH MULTIPLE STRINGS
    ['foo\nbar', 'b', '', ''],
    ['he\nty', 'e', '', 'Multiple messages in text.'],
    # LONGER INAPPROPRIATE
    ['wrrld', 'rl', '', ''],
    ['ddadd', 'da', '', 'Inappropriate word is longer than 1 symbol.'],
    # MULTIPLE CASES
    ['banana', 'a', '', ''],
    ['sdsdf', 's', '', 'Inappropriate word enters message multiple times.'],
    # REPETITIVE PATTERNS
    ['abcdefabcdefabcdef', 'abc,def', '******************', '', True],
    # NUMBER OF PATTERNS > 2
    ['abbbdcdbeeff', 'ab,cd,ef', '**bbd**be**f', '', True],
    # PATTERNS ARE LONG
    ['abcdefghijklmnopqrst', 'abcdefgh,mnopqrst', '********ijkl********', '', True],
    ['aaaaaaaaabbbbbbbbbccccccccc', 'aaaaaaaaa,bbbbbbbbb,ccccccccc', '*' * 27, 'Patterns are long.', True],
    # PATTERNS WITH SUFFIXES EQUAL TO PREFIXES
    ['ababababa', 'aba,bab', '*********', '', True],
    ['abcdabcdabcda', 'abc,cda', '*************', 'Patterns with suffixes equal to prefixes.', True],
    # PATTERNS HAVE SAME CHARACTERS BUT DIFFERENT ORDERS
    ['baabbabab', 'ab,ba', '*********', '', True],
    ['abcbabcbabcbacba', 'abc,cba', '****************', 'Patterns have same characters but different orders.', True],
    # OVERLAPPING PATTERNS
    ['abcdedcdbdcabcd', 'ab,bc,cd', '****ed**bdc****', '', True],
    ['aabbccaabbccaaa', 'aabb,bbcc,ccaa', '**************a', 'Some patterns are overlapping.', True],
    # 2 PATTERNS HAVE SAME PREFIXES
    ['abbadab', 'ab,ad', '**b****', '', True],
    ['abceabcdabcabcd', 'abcd,abce', '********abc****', '', True],
    ['abcdabcdabcdeabcdf', 'abcde,abcdf', 'abcdabcd**********', 'Two patterns have same prefixes.', True],
    # MULTIPLE PATTERNS HAVE SAME PREFIXES
    ['cbcabcac', 'ab,ac,ad', 'cbc**c**', '', True],
    ['abceabcdfacdeabcd', 'abcd,acde,abcf', 'abce****f********', '', True],
    ['abababbabbaabbababa', 'aa,ab,ba,bb', '*******************', '', True],
    ['abcdabcdabcdeabcdf', 'ab,ac,ad,ae', '**cd**cd**cde**cdf', 'Multiple patterns have same prefixes.', True],
    ['aaaaabaababbaaaabaa', 'aaaab,aaaba,aaaaa', '*******ababb******a', 'Multiple patterns have same prefixes.', True],
    ['aabababbabababbabab', 'aaa,aab,aba,abb,baa,bab,bba,bbb', '*' * 19, 'Multiple patterns have same prefixes.', True],
    # VARIOUS PATTERN LENGTHS
    ['abcacbac', 'ab,acb', '**c***ac', '', True],
    ['cadcbcadcbabbbabcdcb', 'abc,cd,bcad,bbb', 'cadc****cba*******cb', '', True],
    ['abcdabcd', 'a,ab,abc,abcd', '********', '', True],
    ['abcdefghij', 'a,bc,def,ghij', '**********', '', True],
    ['bbcde', 'bc,cde', 'b****', 'Various pattern lengths.', True],
    ['abcdcdbcdabcd', 'abcd,bcd,cd,d', '*************', 'Various pattern lengths.', True],
    ['aaaaabaababbaaaabaaaaa', 'aa,aaa,ab,baa', '**********************', 'Various pattern lengths.', True],
]

stage5_specific = [
    # AHO-CORASICK SPECIFIC
    ['the cat in the hat wears a hat, while it is a cat', 'hat,cat,the',
     '*** *** in *** *** wears a ***, while it is a ***', 'Hidden case'],
    ['sher shehers', 'he,she,her,hers', '**** *******', 'Hidden case'],
    ['the present was about prefixes', 'pre,prefix,present', 'the ******* was about ******es', 'Hidden case'],
    ['xyhjklvqet', 'hjkl,vqet', 'xy********', 'Hidden case'],
    ['level', 'eve,ele', 'l***l', 'Hidden case'],
    ['abcdeabcde', 'deabc,cdeab', 'ab******de', 'Hidden case'],
    ['aaaaaaa', 'aaa,aaa', '*******', 'Hidden case'],
    ['xzyzxyyxyyzxzyxyxyzxyyzxzxyzxyyxyzxyzzyxyzyxyzyx', 'xyz,xzy,yxz,yzx,zxy,zyx',
     '******yxy******y**************y*****************', 'Hidden case'],
    ['xzyzxyyxyyzxzyxyxyzxyyzxzxyzxyyxyzxyzzyxyzyxyzyx', 'xyz,xzyyxz,yzxzxy,zyx',
     'xzyzxyyxyyzx***y***xy*******xyy*****************', 'Hidden case'],
    ['a' * 24, ','.join(['a' * i for i in range(1, 24)]), '*' * 24, 'Hidden case'],
    ['ab' * 24, ','.join(['b' * i for i in range(1, 24)]), 'a*' * 24, 'Hidden case'],
    ['abb' * 24, ','.join(['abb' * i for i in range(1, 24)]), '*' * 72, 'Hidden case'],
]

stage6_tests = [
    # SIMPLE TESTS
    ['abcdef', 'b*d', 'a***ef', '', True],
    ['abcdef', 'c*e', 'ab***f', '', True],
    # MULTIPLE ENTRIES IN HAYSTACK
    ['abcdefbcd', 'b*d', 'a***ef***', '', True],
    ['aadaefbcdadaef', 'ad*ef', 'a*****bcd*****', 'Multiple entries in message', True],
    # REPETITIVE PARTS
    ['abcefabcefabcef', 'ab*ef', '*' * 15, '', True],
    # VARIOUS NON-JOKER SEQUENCES LENGTHS
    ['abcdefbfd', 'bc*efb', 'a******fd', '', True],
    ['aadaefbcdadbef', 'adae*bc', 'a*******dadbef', 'Various non-joker sequences lengths', True],
    # MULTIPLE DIFFERENT ENTRIES IN MESSAGE
    ['abcdefbfd', 'b*d', 'a***ef***', '', True],
    ['aadaefbcdadbef', 'ad*ef', 'a*****bcd*****', 'Multiple different entries in message', True],
    # 2 JOKERS IN PATTERN
    ['abcdefbfd', 'b*d*f', 'a*****bfd', '', True],
    ['aadaefbcdadbef', 'bc*ad*e', 'aadaef*******f', 'Two jokers in pattern', True],
    # MULTIPLE JOKERS IN PATTERN
    ['abcdefbad', 'b*d*f*a', 'a*******d', '', True],
    ['aadaefbcdadabefcdaa', 'bc*ada*e*cd', 'aadaef***********aa', 'Multiple jokers in pattern', True],
    # JOKER SEQUENCE IN PATTERN
    ['abcdef', 'b**e', 'a****f', '', True],
    ['addbabdabdbb', 'bab***db', 'add********b', 'Joker sequence in pattern', True],
    # 2 JOKER SEQUENCES IN PATTERN
    ['abcedeffed', 'b**d**f', 'a*******ed', '', True],
    ['aadaefbcedadbacefaa', 'bc**ad***e', 'aadaef**********faa', 'Two joker sequences in pattern', True],
    # MULTIPLE JOKER SEQUENCES IN PATTERN
    ['abcdefffbaad', 'b*d***f**a', 'a**********d', '', True],
    ['bcaabeaccfaaacbecccdaabc', 'b***c*aa***e**cd', 'bcaa****************aabc', 'Multiple jokers in pattern', True],
    # OVERLAPPING PATTERNS
    ['abcbabcba', 'b*b', 'a*******a', '', True],
    ['bacafbfad', 'a**f', 'b******ad', '', True],
    ['baafffccfcdc', 'a**f*c', 'b*******fcdc', 'Overlapping patterns', True],
    # PATTERN ENDS WITH JOKER SEQUENCE
    ['ahelloworldahelloalicea', 'hello*****', 'a**********a**********a', ''],
    ['bdabccdefgbdabffdbacbd', 'ab**d***', 'bd********bd********bd', 'Pattern ends with joker sequence'],
    # PATTERN STARTS WITH JOKER SEQUENCE
    ['aworldhelloalicehelloa', '*****hello', 'a********************a', ''],
    ['bdefgabccdbdbacabffdbd', '***ab**d', 'bd********bd********bd', 'Pattern starts with joker sequence'],
    # PATTERN STARTS AND ENDS WITH JOKER SEQUENCES
    ['bacbcbabb', '*b*', 'ba********', ''],
    ['cbbcbheyacheyhuhd', '***hey***', 'cb**************d', ''],
    ['cbbcbheacyachebcyhuhd', '***he**y***', 'cb******************d', 'Pattern starts and ends with joker sequence'],
    # PATTERN IS JOKER ONLY
    ['fedbca', '*', '******', ''],
    ['abcdefghijklmnop', '***', '****************', ''],
    ['acbdbcad', '********', '********', 'Pattern has only joker symbols'],
    # SIMILAR NON-JOKER SEQUENCES
    ['ababcabcfabcefabc', '***abc*abc**', 'ab************abc', ''],
    ['aaabacadaeafagahaiaj', '*a*a*a*a*', 'a*******************', ''],
    ['abcabcabdbabcabbd', 'abc*bd*abc**', 'abc************bd', 'Pattern has similar non-joker sequences.'],
    # NON-JOKER SEQUENCES HAVE SAME PREFIXES
    ['ababcabcabcdeabcdbd', '***ab*abc**abcd', 'ab***************bd', ''],
    ['baaaabbaccadfaeyafaga', '*aa*ab*ac*ad*ae*', '****************afaga', ''],
    ['heskyitessmesk', 'esk***es**esk', 'h*************', 'Non-joker sequences in pattern have same prefixes'],

    # SPECIFIC HIDDEN CASES
    ['the cat in the hat wears a hat, while it is a cat', '*a*',
     'the *** in the *** w***s******, while it is******', 'Hidden case'],
    ['the presentation was about prefixes', '**e', '*** *****ntation was about ***f***s', 'Hidden case'],
    ['level', 'e*e', 'l***l', 'Hidden case'],
    ['abcdeabcde', '**a**', 'abc*****de', 'Hidden case'],
    ['aaaaaaa', '*a', '*******', 'Hidden case'],
    ['xzyzxyyxyyzxzyxyxyzxyyzxzxyzxyyxyzxyzzyxyzyxyzyx', 'z**',
     'x*****yxyy*****yxy***y********yxy*******y***y***', 'Hidden case'],
    ['ahab' * 100, 'aha*', '*' * 400, 'Hidden case'],
    ['verylong' * 1000, '*', '*' * 8000, 'Hidden case'],
    ['congratulations last testcase', 'congratulations **** ********', '*' * 29, 'Hidden case'],
]


def format1(algorithm):
    return [
        # FORMAT TESTING
        format_test_case('test', 'e', algorithm, False),
        # FORMAT TESTING ON GOOD-TEXT
        format_test_case('test', 'f', algorithm, True),
        # SIMPLE TESTS
        *simple_test_case('test', 'e', 't*st', algorithm, ""),
        # TESTS WITH MULTIPLE STRINGS
        *simple_test_case('foo\nbar', 'b', 'foo\n*ar', algorithm, ""),
        # LONGER INAPPROPRIATE
        *simple_test_case('wrrld', 'rl', 'wr**d', algorithm, ""),
        # MULTIPLE CASES
        *simple_test_case('furfural', 'ur', 'f**f**al', algorithm, ""),
        # CROSSING INAPPROPRIATE
        *simple_test_case('abcbcb', 'bcb', 'a*****', algorithm, ""),
        *simple_test_case('abbabbabba', 'abba', '**********', algorithm,
                          "Inappropriate words overlapping"),
    ]


def general(algorithm):
    return [
        # OTHER CASES AND MORE EXAMPLES
        *simple_test_case('abcdefg', 'fgh', '', algorithm, ""),
        *simple_test_case('foo\nbar', 'oba', '', algorithm,
                          "Inappropriate word between two messages"),
        *simple_test_case('aaaaaaaaaaaaaa', 'aaa', '**************', algorithm,
                          "Inappropriate words overlapping"),
        *simple_test_case('abcde', 'cde', 'ab***', algorithm,
                          "Inappropriate word is in the end of message"),
        *simple_test_case('ccdd' * 10, 'ccddcc', '****' * 9 + '**dd', algorithm,
                          "Multiple letters in inappropriate words overlapping"),
        *simple_test_case('askdjbgfkvbew', 'askdjbgfkvbew', '*************', algorithm,
                          "Inappropriate word takes whole message"),
        *simple_test_case('a' * 100, 'a' * 2, '*' * 100, algorithm,
                          "Long message and short inappropriate word"),
        *simple_test_case('banana', 'apple', '', algorithm,
                          "No inappropriate words in message"),
        *simple_test_case('abc', 'abcdefg', '', algorithm,
                          "Inappropriate word is longer than message"),
    ]


def format2(algorithm):
    return [
        # FORMAT TESTING
        format_test_case('test', 'e,s', algorithm, False),
        # FORMAT TESTING ON GOOD-TEXT
        format_test_case('test', 'f,l', algorithm, True),
        # SIMPLE TESTS
        *simple_test_case('test', 'e,s', 't**t', algorithm, ""),
        # TESTS WITH MULTIPLE STRINGS
        *simple_test_case('foo\nbar', 'f,b', '*oo\n*ar', algorithm, ""),
    ]
