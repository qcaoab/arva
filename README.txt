



-------- Forwarded Message --------
Subject: 	Python code
Date: 	Sat, 18 Sep 2021 12:58:55 -0500
From: 	Pieter van Staden <pieter.vanstaden@gmail.com>
To: 	paforsyt <paforsyt@uwaterloo.ca>, Yuying Li <yuying@uwaterloo.ca>


Hi Peter/Yuying,

I have attached the Python code, and also re-ran this exact code on my laptop to make sure everything works. The outputs will be sent in a separate email.

Couple of things:

    The code has lots of comments. Probably too many. But it shouldn't be too hard to figure out what is going on.

    Only the main code file is run, i.e. with "_MAIN_" in title; it sets all the main parameters. No need to open the other files to get results.
        I've included 2 main code files, as examples, discussed below.

    Running the main code file "_MAINv08_ LiForsyth_Table7_simulated.py" should reproduce the NN results as per Table 7 in your 2019 paper (working paper version).
        Note that this is based on simulated training data, and the simulation happens inside the code. It does not require data in the "Market data" folder.
        I use 500k training data paths, so the MV results I get is basically identical to the "optimal" reported in the table, Mean/Stdev = 705/153.
        Also, slightly different results will be obtained from different runs, since it uses stochastic gradient descent.
        Only training, no testing (this is simulated data).
        Results are obtained using 2 NN hidden layers, just to show how to add additional layers.

    Running the (second) main code file "_MAINv08_ LiForsyth_Table7_but_bootstrapped.py"  is exactly the same setup as the previous one, but now bootstraps the training/testing data based on the historical data.
        These results will *not* agree with your table of course, I just give it as an example of bootstrapping with the same assets.
        All the bootstrap parameters are setup in the code, and the bootstrapping for training/testing happens in the code too.
        Only the time series of monthly, nominal returns are needed for each asset.
        Inflation-adjustments are also done in the code.
        Training (500k, 1926:01-2009:12, exp blk 6 months) + testing (100k, 2010:01-2020:12, exp blk 3 months) are done. Also, implementation on the actual historical path, starting 1980:01, 1985:01, 1990:01.

    Market_data folder:
        Contains the latest CRSP and Factor data updated until the end of 2020.
        I.e. they can also run a multi-asset case (with factors) if needed.

Thanks very much,
Pieter


