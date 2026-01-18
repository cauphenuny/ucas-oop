#show emph: text.with(font: ("New Computer Modern", "STKaiti"))
#set text(font: ("Libertinus Serif", "Songti SC"), lang: "zh")
#show emph: text.with(font: ("Libertinus Serif", "STKaiti"))
#import "@preview/theorion:0.4.1"
#import "@preview/tablem:0.3.0": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *


#import "meta.typ": *
#import "@preview/touying:0.6.1": *
#import "@preview/numbly:0.1.0": *

#show: doc => {
  // import themes.university: *
  // import themes.metropolis: *
  import themes.dewdrop: *
  show: dewdrop-theme.with(
    aspect-ratio: "16-9",
    footer: self => grid(
      columns: (1fr, 1fr, 1fr),
      align: center + horizon,
      self.info.author, self.info.title, self.info.date.display(),
    ),
    navigation: "mini-slides",
    config-info(
      title: meta.slide-title,
      subtitle: meta.subtitle,
      date: meta.date,
      author: meta.author,
    ),
  )
  // show: university-theme.with(
  //   aspect-ratio: "16-9",
  //   footer: self => grid(
  //     columns: (1fr, 1fr, 1fr),
  //     align: center + horizon,
  //     self.info.author,
  //     self.info.title,
  //     self.info.date.display(),
  //     ),
  //   config-info(
  //     title: meta.slide-title,
  //     subtitle: meta.subtitle,
  //   )
  // )
  // show: metropolis-theme.with(
  //   aspect-ratio: "16-9",
  //   footer: self => grid(
  //     columns: (1fr, 1fr, 1fr),
  //     align: center + horizon,
  //     self.info.author,
  //     self.info.title,
  //     self.info.date.display(),
  //     ),
  //   config-info(
  //     title: meta.slide-title,
  //     subtitle: meta.subtitle,
  //     author: meta.author,
  //     date: meta.date,
  //     institution: meta.institution,
  //     logo: none,
  //   ),
  // )
  show: text.with(size: 0.90em)
  show: codly-init.with()
  show raw.where(block: true): text.with(size: 0.8em)

  set heading(numbering: numbly("{1:一}、", default: "1.1  "))

  title-slide()
  doc
  focus-slide[
    Thanks!
  ]
}

#include "main.typ"