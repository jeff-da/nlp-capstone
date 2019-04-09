---
layout: post
title:  "Team Information"
date:   2019-04-08 18:58:37 -0700
---
## Team Name
Facebook fucked up our wedding picture.

## Team Members
Dan Tran, Jack Khuu, and Jeff Da

## Project Ideas

### NLVR2 - Visual QA with Real Images.

NLVR2 is a dataset that focuses on VQA with image pairs. For example, you might have two images where the caption is "there are 8 bottles in total.", and you would need to determine if this is true or false.

#### Minimum Viable Action Plan

First, build baselines for RNN+CNN / MaxEnt. Second, improve the baselines by planning and testing improvements to the failure cases of the RNN+CNN / MaxEnt models. Third, look at the failure and success cases of our new model and implement improvements.

#### Stretch Goals

Beat MaxEnt, improve to a more plausible accuracy number (~60%). Design and test a new model architecture. Try different CNN implementations (skips).

### VCR - Visual Commonsense Reasoning.

VCR is a dataset where you need to give reasoning for why you pick one of four multiple choice answers based off a given image. This reasoning often involves information not implicit in the text.

#### Minimum Viable Action Plan

First, build text-only baselines. Then, add some implementation of a VQA infrastructure (i.e. concat a ResNet on top of the text-only baselines). Then, plan and develop a new model based off the current state of the art. (i.e. R2C). Fine-tune this model (if it works), or figure out why it doesn't work (if it does not :() and fix it.

#### Stretch Goals

Beat text-only baselines. Beat state of the art. Create a new model infrastructure based on past research.

### ReCoRD - Commonsense reading comprehension.

Record is a dataset that focuses on machine reading comprehension requiring commonsense reasoning.

#### Minimum Viable Action Plan

First, reimplement the baselines presented in the paper (DotQA + ELMo/BERT). Next, select test categories where the model fails and see similarities between those. Create a model that attempts to work on these categories where the current state of the art fails.

#### Stretch Goals

Beat or hit current baselines with DocQA and ELMo. Create significant analysis on results. Create a model and infrastructure that improves the baseline.

## Github Project

This project will be hosted at [https://github.com/Dan-Tran/nlp-capstone](https://github.com/Dan-Tran/nlp-capstone).