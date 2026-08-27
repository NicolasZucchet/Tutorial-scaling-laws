<!-- The three stages of the modern LLM training pipeline, one column per stage,
     compared on the same four rows: goal, objective, one line of example data,
     and when the stage became standard practice.  Dates are the public
     milestones: GPT in 2018, ChatGPT in November 2022, o1 in September 2024
     and DeepSeek-R1 in January 2025.  Columns are revealed one at a time. -->
<div class="pipeline" role="img" aria-label="The training pipeline in three stages. Pretraining, from 2018 with GPT, packs as much knowledge as possible into the weights by minimizing the cross-entropy of every next token on raw web text. Instruction fine-tuning, from 2022 with ChatGPT, makes the model useful in interaction by minimizing the cross-entropy of curated responses on prompt-response pairs. Reinforcement learning, from 2024 with o1 and DeepSeek-R1, improves reasoning and coding by maximizing the reward of sampled solutions to problems whose answer can be checked.">

  <div class="pipe-col pipe-pre">
    <div class="pipe-head"><span class="pipe-step">1</span><span class="pipe-title">Pretraining</span></div>
    <div class="pipe-label">Goal</div>
    <div class="pipe-value">pack <strong>as much knowledge as possible</strong> into the weights</div>
    <div class="pipe-label">Objective</div>
    <div class="pipe-value">cross-entropy of the <strong>next token</strong>, on every token</div>
    <div class="pipe-label">Data</div>
    <div class="pipe-data">raw text scraped from the web<br><span class="pipe-mono">&hellip;the cat sat on the mat, and<br>the dog lay by the door&hellip;</span></div>
    <div class="pipe-era">2018+ &#8211; GPT</div>
  </div>

  <div class="pipe-col pipe-sft fragment" data-colloquium-fragment="1">
    <div class="pipe-head"><span class="pipe-step">2</span><span class="pipe-title">Instruction fine-tuning</span></div>
    <div class="pipe-label">Goal</div>
    <div class="pipe-value">make the model <strong>useful in interaction</strong></div>
    <div class="pipe-label">Objective</div>
    <div class="pipe-value">same cross-entropy, but on <strong>curated responses</strong> only</div>
    <div class="pipe-label">Data</div>
    <div class="pipe-data">prompt&#8211;response pairs written by humans<br><span class="pipe-mono">User: why is the sky blue?<br>Assistant: sunlight scatters off air&hellip;</span></div>
    <div class="pipe-era">2022+ &#8211; ChatGPT</div>
  </div>

  <div class="pipe-col pipe-rl fragment" data-colloquium-fragment="1">
    <div class="pipe-head"><span class="pipe-step">3</span><span class="pipe-title">Reinforcement learning</span></div>
    <div class="pipe-label">Goal</div>
    <div class="pipe-value">improve <strong>reasoning and coding</strong></div>
    <div class="pipe-label">Objective</div>
    <div class="pipe-value">maximize the <strong>reward of a correct solution</strong></div>
    <div class="pipe-label">Data</div>
    <div class="pipe-data">problems whose answer can be checked<br><span class="pipe-mono">If 3x + 7 = 22, what is x?<br>grader: is the final answer 5?</span></div>
    <div class="pipe-era">2024+ &#8211; o1, DeepSeek-R1</div>
  </div>

</div>
