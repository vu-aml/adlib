TREC 2005 Spam Track Public Corpus

Copyright 2005 Gordon V. Cormack and Thomas R. Lynam

    gvcormac@uwaterloo.ca
    trlynam@uwaterloo.ca

Includes material released to public pomain and material
used with permission.

Permission is granted for research use by registered
participants in TREC 2005 (trec.nist.gov).  "Research
use" means any bona fide scientific investigation
including, but not restricted to, the 2005 "Spam Track" 
tasks.

Permission is NOT granted to publish this corpus
or any material portion (including file names and
judgements).

It is our intention to make this corpus available
to non-participants, on request, at a future date.


----

INSTRUCTIONS

1. The compressed file may be uncompressed with gzip, Winzip,
   or any other utility that understands gzip format.

2. The compressed file will unpack to a folder named trec05p-1

3. There is one main corpus with four subsets:

   trec05p-1/full   -- the main corpus with 92,189 messages
   trec05p-1/ham25  -- subset of full: 100% of spam, 25% of ham
   trec05p-1/ham50  -- subset of full: 100% of spam, 50% of ham
   trec05p-1/spam25 -- subset of full: 25% of spam, 100% of ham
   trec05p-1/spam50 -- subset of full: 50% of spam, 100% of ham

4. Corpus is compatible with "TREC Spam Filter Evaluation Toolkit"
   using the commands:

      run.sh trec05p-1/full/
      run.sh trec05p-1/ham25/
      run.sh trec05p-1/ham50/
      run.sh trec05p-1/spam25/
      run.sh trec05p-1/spam50/
