# -*- coding: utf-8 -*-
# Python

"""Copyright (c) Alexander Fedotov.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


def decode(human_said, response, recorder=None):
    candidates = response['choices']
    if len(candidates) == 1:
        answer = candidates[0]['message']['content']
        if recorder:
            machine_answered = dict(role='assistant',
                                    content=answer)
            events = [human_said, machine_answered]
            recorder.log_it(events)
            initial_text = human_said['content']
            records = [dict(Human=initial_text), dict(machine=answer)]
            recorder.record_it(records)
        return answer
    elif len(candidates) > 1:
        answers = []
        for candidate in candidates:
            text = candidate['message']['content']
            answers.append(text)
        if recorder:
            # only the first answer is logged (because of continuations)
            # make your own logger if you will be choosing after every turn;
            # I edit records manually and overwrite log with the help of Scribe
            # see the Grammateus package.
            machine_answer = dict(role='assistant', content=answers[0])
            events = [human_said, machine_answer]
            recorder.log_it(events)
            initial_text = human_said['content']
            records = [dict(Human=initial_text), dict(machine=answers)]
            recorder.record_it(records)
        return answers


def encode(records, recorder=None):
    log = []
    for record in records:
        keys = record.keys()
        key = next(iter(record.keys()))
        if key == 'Human':
            user_said = dict(role='user', content=record['Human'])
            log.append(user_said)
        elif key == 'machine':
            text = record['machine']
            if isinstance(text, str):
                utterance = text
            elif isinstance(text, list):
                utterance = text[0]
            else:
                utterance = ''
                print('unknown record type')
            machine_said = dict(role='assistant', content=utterance)
            log.append(machine_said)
        elif key == 'instruction':
            deus_said = dict(role='system', content=record['instruction'])  # θεός
            log.append(deus_said)
        else:
            print('unknown record type')
    if recorder:
        recorder.log_it(log)
    return log
