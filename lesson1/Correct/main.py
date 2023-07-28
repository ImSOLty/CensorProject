import argparse
import re
import additional
from additional import log
from brute_force import brute_force
from kmp import KMP, calculate_lps
from boyer_moore import boyer_moore, bad_character_table, good_suffix_table
from rabin_karp import rabin_karp, hash_table
from aho_corasick import aho_corasick, aho_create_statemachine

DIVIDER = "=" * 58


def output_result(output_type, result, chat):
    if output_type == "identify":
        log("Inappropriate content detected in messages\n" + DIVIDER)
        print("\n".join([chat[i[0]] for i in result]))
    elif output_type == "replace":
        log("Inappropriate content detected and replaced with asterisks\n" + DIVIDER)
        for i in result:
            for j in i[1]:
                chat[i[0]] = chat[i[0]][:j[1]] + "*" * len(j[0]) + chat[i[0]][j[1] + len(j[0]):]
        print("\n".join(chat))
    elif output_type == "remove":
        log("Inappropriate messages removed\n" + DIVIDER)
        ex = [i[0] for i in result]
        print("\n".join([chat[j] for j in range(len(chat)) if j not in ex]))


def preprocessing(alg, words):
    log("Preprocessing:")
    if alg == 'kmp':
        return calculate_lps(words[0])
    elif alg == 'boyer-moore':
        return [bad_character_table(words[0]), good_suffix_table(words[0])]
    elif alg == 'rabin-karp':
        return hash_table(words)
    elif alg == 'aho-corasick':
        return aho_create_statemachine(words)
    else:
        if words[0] == '*' * len(words[0]):
            return None
        matches = [(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'[^\*]+', words[0])]
        word, indices = zip(*matches)
        log("Substrings and indices without wildcards:")
        for i in range(len(word)):
            log(f'\t{word[i]}({indices[i][0]})')
        return aho_create_statemachine(list(word), [i[0] for i in indices])


def main(args):
    with open(args.file) as f:
        chat_list = [line.strip().lower() for line in f]
    words = [args.word]
    if args.multiple:
        with open(args.word) as f:
            words = [line.strip().lower() for line in f]
    result = []
    log(f"Started \"{args.algorithm}\" algorithm")

    log(DIVIDER)
    if args.algorithm in ['kmp', 'boyer-moore', 'rabin-karp', 'aho-corasick', 'aho-corasick-wildcard']:
        helper = preprocessing(args.algorithm, words)
        log(DIVIDER)

    for msg_id in range(len(chat_list)):
        log(f"Line: \"{chat_list[msg_id]}\"")
        in_msg = []
        if args.algorithm == "brute-force":
            in_msg = brute_force(words[0], chat_list[msg_id])
        elif args.algorithm == "kmp":
            in_msg = KMP(words[0], chat_list[msg_id], helper)
        elif args.algorithm == "boyer-moore":
            in_msg = boyer_moore(words[0], chat_list[msg_id], helper)
        elif args.algorithm == "rabin-karp":
            in_msg = rabin_karp(words, chat_list[msg_id], helper)
        elif args.algorithm == "aho-corasick":
            in_msg = aho_corasick(chat_list[msg_id], helper)
        elif args.algorithm == "aho-corasick-wildcard":
            in_msg = aho_corasick(chat_list[msg_id], helper, args.word)
        if len(in_msg) != 0:
            result.append([msg_id, in_msg])

    log(DIVIDER)
    if len(result) == 0:
        log("No inappropriate content detected")
    else:
        output_result(args.type, result, chat_list)


def calc_hash(s):
    hash_value = 0
    for char in s:
        hash_value = (hash_value * 256 + ord(char)) % 101
    return hash_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", type=str,
                        choices=['brute-force', 'kmp', 'boyer-moore', 'rabin-karp', 'aho-corasick',
                                 'aho-corasick-wildcard'],
                        help="String-searching algorithm", required=True)
    parser.add_argument("-w", "--word", type=str, help="Inappropriate content", required=True)
    parser.add_argument("-f", "--file", type=str, help="Text file that should be processed", required=True)
    parser.add_argument("-t", "--type", type=str, choices=['replace', 'remove', 'identify'],
                        help="Process type.", required=True)
    parser.add_argument("-l", "--logging", action='store_true', help="Enable logging")
    parser.add_argument("-m", "--multiple", action='store_true',
                        help="Multiple words from a file provided with --word argument")
    arguments = parser.parse_args()

    if arguments.multiple and (arguments.algorithm not in ['rabin-karp', 'aho-corasick']):
        parser.error("--multiple requires rabin-karp or aho-corasick algorithm usage.")

    additional.logging = arguments.logging
    main(arguments)
