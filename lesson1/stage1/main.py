import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", type=str,
                        choices=['brute-force', 'knuth-morris-pratt', 'boyer-moore', 'rabin-karp', 'aho-corasick',
                                 'aho-corasick-wildcard'],
                        help="String-searching algorithm", required=True)
    parser.add_argument("-w", "--word", type=str, help="Inappropriate content", required=True)
    parser.add_argument("-f", "--file", type=str, help="Text file that should be processed", required=True)
    parser.add_argument("-l", "--logging", action='store_true', help="Enable logging")
    parser.add_argument("-m", "--multiple", action='store_true',
                        help="Multiple words from a file provided with --word argument")
    arguments = parser.parse_args()

    if arguments.multiple and (arguments.algorithm not in ['rabin-karp', 'aho-corasick']):
        parser.error("--multiple requires rabin-karp or aho-corasick algorithm usage.")
