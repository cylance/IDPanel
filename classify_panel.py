from gevent.monkey import patch_all
patch_all()
from gevent.pool import Pool
from os.path import isfile
from idpanel.classification import ClassificationEngine
from idpanel.utility import make_request
from sys import stderr, stdin


def get_result_wrapper((base_url, request)):
    try:
        url = base_url + request
        code, ssdeep = make_request(url, True)
        #stderr.write(repr((url, code, ssdeep)) + "\n")
        return base_url, request, {"code": code, "content_ssdeep": ssdeep}
    except:
        return None, None, None


def reformat_url(url):
    if url[-1] != "/":
        url += "/"

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url

    return url


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Path to model on disk")
    parser.add_argument('url', type=str, help="Base url to check, or path to file to read (- for stdin)")

    args = parser.parse_args()

    base_url = args.url
    base_urls = []

    if isfile(base_url):
        # read file for urls
        with open(base_url, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                line = reformat_url(line)
                if line not in base_urls:
                    base_urls.append(line)

    elif base_url == "-":
        # read from stdin
        for line in stdin:
            line = line.strip()
            if len(line) == 0:
                continue
            line = reformat_url(line)
            if line not in base_urls:
                base_urls.append(line)

    else:
        # its probably a url...
        base_url = reformat_url(base_url)
        base_urls = [base_url]

    model_path = args.model

    classifier = ClassificationEngine.load_model(model_path)
    pool = Pool(size=16)

    offsets = classifier.get_required_requests()
    results = {}

    stderr.write("Identifying panels we can actually reach\n")
    for base_url, r1, r2 in pool.imap_unordered(get_result_wrapper, [(i, "") for i in base_urls]):
        if base_url is not None:
            stderr.write("We can reach {0}\n".format(base_url))
            results[base_url] = {}

    requests_to_make = []
    for offset in offsets:
        for base_url in results.keys():
            requests_to_make.append((base_url, offset))

    stderr.write("Making {0} total requests to {1} servers\n".format(len(requests_to_make), len(results.keys())))
    for base_url, request, result in pool.imap_unordered(get_result_wrapper, requests_to_make):
        if base_url is None:
            continue
        results[base_url][request] = result

    for base_url in results.keys():
        label, scores, label_scores = classifier.get_label_probs(results[base_url])
        print "\t".join([label, base_url, repr(label_scores)])
