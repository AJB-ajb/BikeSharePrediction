#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [A spatio-temporal graph neural network approach for bike-sharing demand prediction],
  abstract: [
    We adapt and analyze the model spatio-temporal graph attention network (ST-GAT) architecture known from traffic prediction to predict bike-sharing windows and compare it with several baselines.
    // analyze scaling behavior of the model (?)
    // analyze transformer variation (?)
    // 
  ],
  authors: (
    (
      name: "Alexander Julius Busch",
      // department: [],
      organization: [University of Hamburg],
      location: [Hamburg, Germany],
      //email: "alexander.busch@"
    ),
    (
      name: "",
      department: [],
      organization: [],
      location: [],
      email: ""
    ),
  ),
  index-terms: ("AI", "Smart Mobility", "GNN", "Bike-Sharing"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)


= Introduction
// summarize the importance of the problem you are trying to solve and the reason that motivated you to select this project. Explain what was the problem or challenge that you were given? state the purpose of the project and how did you solve it? Enumerate the objectives of the project and describe in brief the structure of the article. 
= Literature Review
// conduct a critical survey on similar solutions and explain how your solution extends or differs from these solutions. 

= Problem Formulation and Modeling
// describe the include mathematical formulation of the problem and possible modeling approach.
= Proposed Solution
//: describe the bio-inspired algorithms selected to solve the project problem.
= Performance Evaluation
//: Establish a set of evaluation metrics and run some experiments with different values of algorithm parameters to quantitativelyand qualitatively assess the performance of the developed solution. Students must identify the pros and cons of each technique and assess the quality of work as well as its fit with project objectives.
= Conclusions & Recommendations
//: summarize the conclusion and future improvement. Explain how did you solve the problem; what problems were met? what did the results show? And how to refine the proposed solution? You may organize ideas using lists or numbered points, if appropriate, but avoid making your article into a check-list or a series of encrypted notes.
= Code <sec:code>
// : provide a GitHub permanent link to the code that implements the proposed solution.

#figure(
  caption: [The Planets of the Solar System and Their Average Distance from the Sun],
  placement: top,
  table(
    // Table styling is not mandated by the IEEE. Feel free to adjust these
    // settings and potentially move them into a set rule.
    columns: (6em, auto),
    align: (left, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },

    table.header[Planet][Distance (million km)],
    [Mercury], [57.9],
    [Venus], [108.2],
    [Earth], [149.6],
    [Mars], [227.9],
    [Jupiter], [778.6],
    [Saturn], [1,433.5],
    [Uranus], [2,872.5],
    [Neptune], [4,495.1],
  )
) <tab:planets>
