# Jeff, Jack, Dan NLP Team

## 8/9. Viable plans.

### Dataset. NLVR2. Visual QA with real images.

**Plan.** First, build baselines for RNN+CNN / MaxEnt. Second, improve the baselines by planning and testing improvements to the failure cases of the RNN+CNN / MaxEnt models. Third, look at the failure and success cases of our new model and implement improvements.

**Goals.** Beat MaxEnt, improve to a more plausible accuracy number (~60%). Design and test a new model architecture. Try different CNN implementations (skips).

### Dataset. VCR. Visual Commonsense Reasoning.

**Plan.** First, build text-only baselines. Then, add some implementation of a VQA infrastructure (i.e. concat a ResNet on top of the text-only baselines). Then, plan and develop a new model based off the current state of the art. (i.e. R2C). Fine-tune this model (if it works), or figure out why it doesn't work (if it does not :() and fix it.

**Goals.** Beat text-only baselines. Beat state of the art. Create a new model infrastructure based on past research.

### Dataset. ReCoRD. Commonsense reading comprehension.

**Plan.** First, reimplement the baselines presented in the paper (DotQA + ELMo/BERT). Next, select test categories where the model fails and see similarities between those. Create a model that attempts to work on these categories where the current state of the art fails.

**Goals.** Beat or hit current baselines with DocQA and ELMo. Create significant analysis on results. Create a model and infrastructure that improves the baseline.
