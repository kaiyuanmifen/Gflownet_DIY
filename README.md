# Gflownet_DIY

A simplest implementation of Glownet with flow matching loss (see https://arxiv.org/abs/2106.04399)

The enviroment of adding integers to different position of an vector (initally vectors of zeros) with a multi-mode reward is included. 

The DAG is acyclic and not a tree. 

This code has been tested and intended to be used as a tutorial for GFN 

If you like ot use a computationlly efficient version or a version with detailed matching or tracjectory matching please refer to our team repo:https://github.com/GFNOrg


FlowFunction.py contain the flow estimation function to output log(flow)

MultiModalSeq.py is the environment

just run python Run.py to train the Gflownets


