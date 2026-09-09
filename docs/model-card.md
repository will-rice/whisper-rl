---
library_name: transformers
license: mit
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-tiny
datasets:
  - mozilla-foundation/common_voice_26_0
language:
  - af
  - am
  - ar
  - as
  - az
  - ba
  - be
  - bg
  - bn
  - br
  - ca
  - cs
  - cy
  - da
  - de
  - el
  - en
  - es
  - et
  - eu
  - fa
  - fi
  - fr
  - gl
  - ha
  - he
  - hi
  - ht
  - hu
  - hy
  - id
  - is
  - it
  - ja
  - ka
  - kk
  - km
  - ko
  - lo
  - lt
  - lv
  - mk
  - ml
  - mn
  - mr
  - ms
  - mt
  - ne
  - nl
  - nn
  - oc
  - pa
  - pl
  - ps
  - pt
  - ro
  - ru
  - sd
  - sk
  - sl
  - sq
  - sr
  - sv
  - sw
  - ta
  - te
  - tg
  - th
  - tk
  - tr
  - tt
  - uk
  - ur
  - uz
  - vi
  - yi
  - yo
  - yue
  - zh
tags:
  - whisper
  - grpo
  - reinforcement-learning
  - asr
  - multilingual
metrics:
  - wer
  - cer
model-index:
  - name: whisper-tiny-grpo
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: overall
          split: test
        metrics:
          - type: wer
            value: 0.6226
            name: WER (overall)
          - type: cer
            value: 0.2789
            name: CER (overall)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: af
          split: test
        metrics:
          - type: wer
            value: 0.8896
            name: WER (af)
          - type: cer
            value: 0.4220
            name: CER (af)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: am
          split: test
        metrics:
          - type: wer
            value: 0.9059
            name: WER (am)
          - type: cer
            value: 0.6430
            name: CER (am)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ar
          split: test
        metrics:
          - type: wer
            value: 0.9494
            name: WER (ar)
          - type: cer
            value: 0.3204
            name: CER (ar)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: as
          split: test
        metrics:
          - type: wer
            value: 0.5186
            name: WER (as)
          - type: cer
            value: 0.2943
            name: CER (as)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: az
          split: test
        metrics:
          - type: wer
            value: 0.8996
            name: WER (az)
          - type: cer
            value: 0.6863
            name: CER (az)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ba
          split: test
        metrics:
          - type: wer
            value: 0.6061
            name: WER (ba)
          - type: cer
            value: 0.2066
            name: CER (ba)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: be
          split: test
        metrics:
          - type: wer
            value: 0.6098
            name: WER (be)
          - type: cer
            value: 0.1651
            name: CER (be)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: bg
          split: test
        metrics:
          - type: wer
            value: 0.6054
            name: WER (bg)
          - type: cer
            value: 0.2055
            name: CER (bg)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: bn
          split: test
        metrics:
          - type: wer
            value: 0.5108
            name: WER (bn)
          - type: cer
            value: 0.3422
            name: CER (bn)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: br
          split: test
        metrics:
          - type: wer
            value: 0.6957
            name: WER (br)
          - type: cer
            value: 0.3180
            name: CER (br)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ca
          split: test
        metrics:
          - type: wer
            value: 0.4281
            name: WER (ca)
          - type: cer
            value: 0.1578
            name: CER (ca)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: cs
          split: test
        metrics:
          - type: wer
            value: 0.7746
            name: WER (cs)
          - type: cer
            value: 0.2804
            name: CER (cs)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: cy
          split: test
        metrics:
          - type: wer
            value: 0.6336
            name: WER (cy)
          - type: cer
            value: 0.2301
            name: CER (cy)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: da
          split: test
        metrics:
          - type: wer
            value: 0.6898
            name: WER (da)
          - type: cer
            value: 0.2884
            name: CER (da)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: de
          split: test
        metrics:
          - type: wer
            value: 0.3517
            name: WER (de)
          - type: cer
            value: 0.1126
            name: CER (de)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: el
          split: test
        metrics:
          - type: wer
            value: 0.5928
            name: WER (el)
          - type: cer
            value: 0.2559
            name: CER (el)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: en
          split: test
        metrics:
          - type: wer
            value: 0.4124
            name: WER (en)
          - type: cer
            value: 0.2110
            name: CER (en)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: es
          split: test
        metrics:
          - type: wer
            value: 0.2786
            name: WER (es)
          - type: cer
            value: 0.1017
            name: CER (es)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: et
          split: test
        metrics:
          - type: wer
            value: 0.7054
            name: WER (et)
          - type: cer
            value: 0.1997
            name: CER (et)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: eu
          split: test
        metrics:
          - type: wer
            value: 0.4465
            name: WER (eu)
          - type: cer
            value: 0.0935
            name: CER (eu)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: fa
          split: test
        metrics:
          - type: wer
            value: 0.7909
            name: WER (fa)
          - type: cer
            value: 0.2856
            name: CER (fa)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: fi
          split: test
        metrics:
          - type: wer
            value: 0.5438
            name: WER (fi)
          - type: cer
            value: 0.1198
            name: CER (fi)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: fr
          split: test
        metrics:
          - type: wer
            value: 0.4385
            name: WER (fr)
          - type: cer
            value: 0.1833
            name: CER (fr)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: gl
          split: test
        metrics:
          - type: wer
            value: 0.3402
            name: WER (gl)
          - type: cer
            value: 0.1039
            name: CER (gl)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ha
          split: test
        metrics:
          - type: wer
            value: 0.6937
            name: WER (ha)
          - type: cer
            value: 0.2691
            name: CER (ha)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: he
          split: test
        metrics:
          - type: wer
            value: 0.7096
            name: WER (he)
          - type: cer
            value: 0.2729
            name: CER (he)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: hi
          split: test
        metrics:
          - type: wer
            value: 0.2445
            name: WER (hi)
          - type: cer
            value: 0.1119
            name: CER (hi)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ht
          split: test
        metrics:
          - type: wer
            value: 0.9474
            name: WER (ht)
          - type: cer
            value: 0.4279
            name: CER (ht)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: hu
          split: test
        metrics:
          - type: wer
            value: 0.6928
            name: WER (hu)
          - type: cer
            value: 0.1889
            name: CER (hu)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: hy-AM
          split: test
        metrics:
          - type: wer
            value: 0.7130
            name: WER (hy-AM)
          - type: cer
            value: 0.2266
            name: CER (hy-AM)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: id
          split: test
        metrics:
          - type: wer
            value: 0.3413
            name: WER (id)
          - type: cer
            value: 0.1192
            name: CER (id)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: is
          split: test
        metrics:
          - type: wer
            value: 0.4399
            name: WER (is)
          - type: cer
            value: 0.3790
            name: CER (is)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: it
          split: test
        metrics:
          - type: wer
            value: 0.4121
            name: WER (it)
          - type: cer
            value: 0.1244
            name: CER (it)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ja
          split: test
        metrics:
          - type: wer
            value: 1.0000
            name: WER (ja)
          - type: cer
            value: 0.5412
            name: CER (ja)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ka
          split: test
        metrics:
          - type: wer
            value: 0.7982
            name: WER (ka)
          - type: cer
            value: 0.3793
            name: CER (ka)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: kk
          split: test
        metrics:
          - type: wer
            value: 0.6856
            name: WER (kk)
          - type: cer
            value: 0.3611
            name: CER (kk)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: km
          split: test
        metrics:
          - type: wer
            value: 1.0000
            name: WER (km)
          - type: cer
            value: 1.3959
            name: CER (km)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ko
          split: test
        metrics:
          - type: wer
            value: 1.1655
            name: WER (ko)
          - type: cer
            value: 0.7311
            name: CER (ko)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: lo
          split: test
        metrics:
          - type: wer
            value: 0.9444
            name: WER (lo)
          - type: cer
            value: 0.8442
            name: CER (lo)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: lt
          split: test
        metrics:
          - type: wer
            value: 0.9078
            name: WER (lt)
          - type: cer
            value: 0.2279
            name: CER (lt)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: lv
          split: test
        metrics:
          - type: wer
            value: 0.5594
            name: WER (lv)
          - type: cer
            value: 0.1601
            name: CER (lv)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: mk
          split: test
        metrics:
          - type: wer
            value: 0.5322
            name: WER (mk)
          - type: cer
            value: 0.1722
            name: CER (mk)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ml
          split: test
        metrics:
          - type: wer
            value: 0.5539
            name: WER (ml)
          - type: cer
            value: 0.3974
            name: CER (ml)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: mn
          split: test
        metrics:
          - type: wer
            value: 0.8295
            name: WER (mn)
          - type: cer
            value: 0.3667
            name: CER (mn)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: mr
          split: test
        metrics:
          - type: wer
            value: 0.4556
            name: WER (mr)
          - type: cer
            value: 0.2433
            name: CER (mr)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ms
          split: test
        metrics:
          - type: wer
            value: 0.9806
            name: WER (ms)
          - type: cer
            value: 0.3127
            name: CER (ms)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: mt
          split: test
        metrics:
          - type: wer
            value: 0.8758
            name: WER (mt)
          - type: cer
            value: 0.4136
            name: CER (mt)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ne-NP
          split: test
        metrics:
          - type: wer
            value: 0.4524
            name: WER (ne-NP)
          - type: cer
            value: 0.2438
            name: CER (ne-NP)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: nl
          split: test
        metrics:
          - type: wer
            value: 0.5019
            name: WER (nl)
          - type: cer
            value: 0.2114
            name: CER (nl)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: nn-NO
          split: test
        metrics:
          - type: wer
            value: 0.6907
            name: WER (nn-NO)
          - type: cer
            value: 0.2715
            name: CER (nn-NO)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: oc
          split: test
        metrics:
          - type: wer
            value: 0.7206
            name: WER (oc)
          - type: cer
            value: 0.2569
            name: CER (oc)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: pa-IN
          split: test
        metrics:
          - type: wer
            value: 0.4205
            name: WER (pa-IN)
          - type: cer
            value: 0.2304
            name: CER (pa-IN)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: pl
          split: test
        metrics:
          - type: wer
            value: 0.4538
            name: WER (pl)
          - type: cer
            value: 0.1405
            name: CER (pl)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ps
          split: test
        metrics:
          - type: wer
            value: 0.7278
            name: WER (ps)
          - type: cer
            value: 0.2955
            name: CER (ps)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: pt
          split: test
        metrics:
          - type: wer
            value: 0.5804
            name: WER (pt)
          - type: cer
            value: 0.2587
            name: CER (pt)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ro
          split: test
        metrics:
          - type: wer
            value: 0.4552
            name: WER (ro)
          - type: cer
            value: 0.1088
            name: CER (ro)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ru
          split: test
        metrics:
          - type: wer
            value: 0.4389
            name: WER (ru)
          - type: cer
            value: 0.1287
            name: CER (ru)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: sd
          split: test
        metrics:
          - type: wer
            value: 1.0553
            name: WER (sd)
          - type: cer
            value: 0.5162
            name: CER (sd)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: sk
          split: test
        metrics:
          - type: wer
            value: 0.6286
            name: WER (sk)
          - type: cer
            value: 0.1680
            name: CER (sk)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: sl
          split: test
        metrics:
          - type: wer
            value: 0.5655
            name: WER (sl)
          - type: cer
            value: 0.1572
            name: CER (sl)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: sq
          split: test
        metrics:
          - type: wer
            value: 0.8477
            name: WER (sq)
          - type: cer
            value: 0.3047
            name: CER (sq)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: sr
          split: test
        metrics:
          - type: wer
            value: 0.5991
            name: WER (sr)
          - type: cer
            value: 0.2412
            name: CER (sr)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: sv-SE
          split: test
        metrics:
          - type: wer
            value: 0.6360
            name: WER (sv-SE)
          - type: cer
            value: 0.3072
            name: CER (sv-SE)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: sw
          split: test
        metrics:
          - type: wer
            value: 0.5259
            name: WER (sw)
          - type: cer
            value: 0.1214
            name: CER (sw)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ta
          split: test
        metrics:
          - type: wer
            value: 0.2964
            name: WER (ta)
          - type: cer
            value: 0.1483
            name: CER (ta)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: te
          split: test
        metrics:
          - type: wer
            value: 0.5055
            name: WER (te)
          - type: cer
            value: 0.2827
            name: CER (te)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: tg
          split: test
        metrics:
          - type: wer
            value: 0.8614
            name: WER (tg)
          - type: cer
            value: 0.2619
            name: CER (tg)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: th
          split: test
        metrics:
          - type: wer
            value: 0.6227
            name: WER (th)
          - type: cer
            value: 0.2580
            name: CER (th)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: tk
          split: test
        metrics:
          - type: wer
            value: 0.8451
            name: WER (tk)
          - type: cer
            value: 0.3308
            name: CER (tk)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: tr
          split: test
        metrics:
          - type: wer
            value: 0.6446
            name: WER (tr)
          - type: cer
            value: 0.2155
            name: CER (tr)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: tt
          split: test
        metrics:
          - type: wer
            value: 0.6000
            name: WER (tt)
          - type: cer
            value: 0.1519
            name: CER (tt)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: uk
          split: test
        metrics:
          - type: wer
            value: 0.6569
            name: WER (uk)
          - type: cer
            value: 0.1960
            name: CER (uk)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: ur
          split: test
        metrics:
          - type: wer
            value: 0.5000
            name: WER (ur)
          - type: cer
            value: 0.2022
            name: CER (ur)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: uz
          split: test
        metrics:
          - type: wer
            value: 0.6552
            name: WER (uz)
          - type: cer
            value: 0.1918
            name: CER (uz)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: vi
          split: test
        metrics:
          - type: wer
            value: 0.6749
            name: WER (vi)
          - type: cer
            value: 0.4004
            name: CER (vi)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: yi
          split: test
        metrics:
          - type: wer
            value: 1.0242
            name: WER (yi)
          - type: cer
            value: 0.6180
            name: CER (yi)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: yo
          split: test
        metrics:
          - type: wer
            value: 0.7660
            name: WER (yo)
          - type: cer
            value: 0.4225
            name: CER (yo)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: yue
          split: test
        metrics:
          - type: wer
            value: 0.8553
            name: WER (yue)
          - type: cer
            value: 0.2432
            name: CER (yue)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: zh-CN
          split: test
        metrics:
          - type: wer
            value: 0.9863
            name: WER (zh-CN)
          - type: cer
            value: 0.3070
            name: CER (zh-CN)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: zh-HK
          split: test
        metrics:
          - type: wer
            value: 1.1818
            name: WER (zh-HK)
          - type: cer
            value: 0.4591
            name: CER (zh-HK)
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: mozilla-foundation/common_voice_26_0
          name: Common Voice 26.0
          config: zh-TW
          split: test
        metrics:
          - type: wer
            value: 0.7333
            name: WER (zh-TW)
          - type: cer
            value: 0.2335
            name: CER (zh-TW)
---

# whisper-tiny-grpo

`openai/whisper-tiny` finetuned with **GRPO** (Group Relative Policy Optimization)
using **word error rate as the reward**, across 81 Common Voice locales (79 distinct Whisper languages).

Instead of cross-entropy against a single reference, the model samples a group of
candidate transcriptions per clip, scores each by its WER against the ground truth,
and is nudged toward the lower-error candidates with a policy-gradient objective
regularized by a KL penalty to the original model. Training code:
[will-rice/whisper-rl](https://github.com/will-rice/whisper-rl).

## Results

Evaluated on the **Common Voice 26.0 test split** — 4,800 clips
round-robin across all 81 locales in the index, greedy decoding, corpus-level rates.

|                       | WER         | CER         |
| --------------------- | ----------- | ----------- |
| `openai/whisper-tiny` | 0.9718      | 0.5836      |
| **this model**        | **0.6226**  | **0.2789**  |
| change                | **-0.3492** | **-0.3047** |

Improves **69 of 81 locales**.

Rates are bucketed by Common Voice locale, not by Whisper language. The two differ: `zh-CN`, `zh-HK` and `zh-TW` are separate locales that all prompt with the same `<|zh|>` token, and regional codes like `pa-IN`, `ne-NP` and `sv-SE` map to `<|pa|>`, `<|ne|>` and `<|sv|>`. 81 locales span 79 Whisper languages.

### Largest gains

| locale | baseline WER | this model | change |
| ------ | ------------ | ---------- | ------ |
| kk     | 2.164        | 0.686      | -1.479 |
| tk     | 1.962        | 0.845      | -1.117 |
| pa-IN  | 1.450        | 0.420      | -1.029 |
| fa     | 1.612        | 0.791      | -0.821 |
| tt     | 1.373        | 0.600      | -0.773 |
| hi     | 0.996        | 0.244      | -0.751 |
| ka     | 1.538        | 0.798      | -0.740 |
| sd     | 1.794        | 1.055      | -0.738 |
| uz     | 1.323        | 0.655      | -0.667 |
| ne-NP  | 1.099        | 0.452      | -0.646 |

Baseline WER above 1.0 means the model emitted more tokens than the reference
contains — runaway insertion, not merely inaccurate transcription. The largest
gains are where that behaviour was worst.

### Regressions (11 of 81 locales)

| locale | baseline WER | this model | change |
| ------ | ------------ | ---------- | ------ |
| ko     | 0.723        | 1.166      | +0.442 |
| ms     | 0.665        | 0.981      | +0.316 |
| ar     | 0.741        | 0.949      | +0.209 |
| en     | 0.308        | 0.412      | +0.105 |
| vi     | 0.614        | 0.675      | +0.061 |
| ru     | 0.389        | 0.439      | +0.050 |
| he     | 0.689        | 0.710      | +0.021 |
| sv-SE  | 0.618        | 0.636      | +0.018 |

The regressions cluster in higher-resource languages where `whisper-tiny` was
already reasonable. The method trades some high-resource accuracy for large
low-resource gains.

## Evaluation methodology

Held-out in both senses that matter:

- **Disjoint from training data.** This checkpoint trained on Common Voice 22;
  evaluation is on Common Voice 26 test. Common Voice partitions speaker-disjoint
  and keeps assignments stable across releases — intersecting CV22 `train` with
  CV26 `test` over six locales found **1 shared speaker in 4,900**.
- **Disjoint from checkpoint selection.** Training selects checkpoints on
  validation reward. The maximum of a noisy sequence of validation passes is
  optimistically biased, so validation numbers overstate. Every figure here is
  from `test`, which nothing in training touched.

Baseline and finetuned models score the **same clips in the same order** — the
evaluation slice is materialized with `take()` and never shuffled — so a
difference can only come from the model.

## Limitations

- Roughly 59 clips per locale at this sample count. The overall figures are
  solid; any single low-resource locale's rate is noisy.
- Compared only against `openai/whisper-tiny`, not against supervised finetuning
  on the same data. This shows GRPO beats the base model, not that it beats SFT.
- Three checkpoints from the same training line were scored, and this one was
  best on test. Selecting the best of three on the evaluation set slightly
  flatters this specific number — though all three beat the baseline by a similar
  margin (WER −0.298 to −0.349), so the improvement itself does not depend on
  which was picked.
- CER is the more meaningful figure for languages without word boundaries
  (`ja`, `th`, `zh-*`).

## Usage

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("wrice/whisper-tiny-grpo")
processor = WhisperProcessor.from_pretrained("wrice/whisper-tiny-grpo")
```

The language token must be pinned at generation time; Whisper's own detection
mislabels lower-resource clips, which is what the training setup assumes.
