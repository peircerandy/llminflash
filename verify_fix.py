import argparse
from chat import chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='predictor')
    parser.add_argument('--top_k', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--window', type=int, default=5)
    
    # We will simulate a user input to check for coherence
    import sys
    from io import StringIO
    sys.stdin = StringIO("What is the capital of France?\nexit\n")
    
    chat(parser.parse_args())
