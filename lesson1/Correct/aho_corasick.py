from additional import log

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

    log("Trie:\n" + root.print())
    return root


def aho_corasick(msg, root, joker=""):
    if joker != '' and joker == '*' * len(joker):
        return [[joker, i] for i in range(len(msg) - len(joker) + 1)]
    c = [0] * len(msg)
    result = []
    node = root
    log(f"|{msg}\n|")
    for i in range(len(msg)):
        while node is not None and msg[i] not in node.goto:
            log(f"\tTransition: \'{node.s if node is not None else ''}\'->"
                f"\'{node.fail.s if node.fail is not None else ''}\'")
            node = node.fail
            if node is not None and node.fail is not None:
                log("|" + msg)
                log("|" + " " * (i - node.lvl) + '^' * node.lvl)
        if node is None:
            node = root
            continue
        log(f"\tTransition: \'{node.s if node is not None else ''}\'->"
            f"\'{node.goto[msg[i]].s if node.goto[msg[i]] is not None else ''}\'")
        node = node.goto[msg[i]]
        log("|" + msg)
        log("|" + " " * (i - node.lvl + 1) + '^' * node.lvl)
        for pattern in node.out:
            if joker != '' and i - len(pattern[0]) - pattern[1] + 1 >= 0:
                c[i - len(pattern[0]) - pattern[1] + 1] += 1
            result.append([pattern[0], i - len(pattern[0]) + 1])
            log(f"\tFound on {i - len(pattern[0]) + 1} position ({pattern[0]})")
    if joker != "":
        result = []
        log(f"C array: {c}")
        for i in range(len(c)):
            if c[i] == len(list(filter(None, joker.split('*')))):
                result.append([joker, i])
                log(f"Pattern ({joker}) found on: {i} position")
    return result
