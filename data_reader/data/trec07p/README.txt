TREC 2007 Public Corpus

Use with the TREC 2007 spam evaluation toolkit.

The corpus trec07p contains 75,419 messages:

    25220 ham
    50199 spam

These messages constitute all the messages delivered to a particular
server between these dates:

    Sun, 8 Apr 2007 13:07:21 -0400
    Fri, 6 Jul 2007 07:04:53 -0400

There are three subcorpora:

trec07p/full/    -  immmediate, full feedback
trec07p/delay/   -  feedback only for first 10,000 messages
trec07p/partial/ -  feedback only for 30,388 messages correponding to 1 recipient

For TREC 2007, please submit 4 runs per filter, with the appropriate run-id
prefix (ffff, the run-id of the filter) and run-id suffix (see below).

runid         command

ffffpf        run.sh trec07p/full/
ffffpd        run.sh trec07p/delay/
ffffpp        run.sh trec07p/partial/
ffffp1000     run07.sh trec07p/full/ ffffpa resultfile 1000

Note:  to create ffffp1000, you must compile the new-for-2007
version of the run shell, run07.sh, which you can build from
the run.activeLearning.cpp C++ source referenced here:

   http://plg.uwaterloo.ca/~gvcormac/spam/onlineActiveIntro.html

Submission deadline is August 22, 2007.
