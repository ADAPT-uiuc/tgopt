#!/usr/bin/env bash
#
# Script to download raw dataset files.

name="$1"
if [[ "$name" == "" ]]; then
  echo "please specify dataset name"
  exit 1
fi

get_snap_file() {
  mkdir -p data
  curl -L -o "data/$1" "$2"
}

get_snap_gzip() {
  mkdir -p data
  curl -L -o "data/$1" "$2"
  gzip -d "data/$1"
}

case "$name" in
  jodie-lastfm)
    get_snap_file 'lastfm.csv' 'http://snap.stanford.edu/jodie/lastfm.csv'
    ;;
  jodie-mooc)
    get_snap_file 'jodie-mooc.csv' 'http://snap.stanford.edu/jodie/mooc.csv'
    ;;
  jodie-reddit)
    get_snap_file 'jodie-reddit.csv' 'http://snap.stanford.edu/jodie/reddit.csv'
    ;;
  jodie-wiki)
    get_snap_file 'jodie-wiki.csv' 'http://snap.stanford.edu/jodie/wikipedia.csv'
    ;;
  snap-email)
    get_snap_gzip 'email-eu-temporal.txt.gz' 'http://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz'
    ;;
  snap-msg)
    get_snap_gzip 'college-msg.txt.gz' 'http://snap.stanford.edu/data/CollegeMsg.txt.gz'
    ;;
  snap-reddit)
    get_snap_file 'reddit-hyperlinks-title.tsv' 'http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv'
    get_snap_file 'reddit-hyperlinks-body.tsv' 'http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv'
    ;;
  *)
    echo "dataset not yet supported: $name"
    ;;
esac
