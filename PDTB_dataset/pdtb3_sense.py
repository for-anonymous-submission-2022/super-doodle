"""
Label | EDSC | NDSC
Expansion.Conjunction       & R & C    & R & C    & N/A
Comparison.Concession       & R & C    & R & C    & N/A
Contingency.Cause           & R & C    & R & C    & N/A
Temporal.Asynchronous       & R & C    & R & C    & N/A
Temporal.Synchronous        & R & C    & R        & N/A
Contingency.Condition       & R & C    & R        & N/A
Comparison.Contrast         & R & C    & R & C    & N/A
Expansion.Manner            & R & C    & R & C    & N/A
Contingency.Purpose         & R & C    & R & C    & N/A
Expansion.Instantiation     & R        & R & C    & N/A
Expansion.Level-of-detail   & R        & R & C    & N/A
Expansion.Substitution      & R        & R        & N/A
Expansion.Disjunction       & R        & NotCon   & N/A
Contingency.Neg-condition   & R        & NotCon   & N/A
Comparison.Similarity       & NotCon   & NotCon   & N/A
Contingency.Cond+SpeechAct  & NotCon   & NotCon   & N/A
Contingency.Cause+Belief    & NotCon   & R        & N/A
Expansion.Exception         & NotCon   & NotCon   & N/A
Expansion.Equivalence       & NotCon   & R        & N/A
Comparison.Conc+SpeechAct   & NotCon   & NotCon   & N/A
Contingency.Cause+SpeechAct & NotCon   & NotCon   & N/A
Contingency.Negative-cause  & NotCon   & NotCon   & N/A
EntRel                      & N/A      & R & C    & N/A
NotCon                      & C        & C        & N/A
"""

def defined_sets(set_name):
    if set_name == "NON_EXPLICIT_RELATIONS":
        return frozenset([
            'Implicit',
            'EntRel',
            'AltLex',
            'AltLexC'
            ])

    if set_name == "EXPLICIT_RELATIONS":
        return frozenset([
            'Explicit',
            ])

    if set_name == "ALL_RELATIONS":
        return frozenset([
            'Explicit',
            'Implicit',
            'EntRel',
            'AltLex',
            'AltLexC'
            ])

    if set_name == "considered_all+EntRel":
        return frozenset([
            'Expansion.Conjunction','Comparison.Concession','Contingency.Cause','Temporal.Asynchronous','Temporal.Synchronous','Contingency.Condition','Comparison.Contrast','Expansion.Manner','Contingency.Purpose','Expansion.Instantiation','Expansion.Level-of-detail','Expansion.Substitution','Expansion.Disjunction','Contingency.Neg-condition','Comparison.Similarity','Contingency.Cond+SpeechAct','Contingency.Cause+Belief','Expansion.Exception','Expansion.Equivalence','Comparison.Conc+SpeechAct','Contingency.Cause+SpeechAct','Contingency.Negative-cause', 'EntRel'
            ])

    if set_name == "notconsidered_all+EntRel":
        return frozenset([
            ])

    if set_name == "considered_exp-L2-14way":
        return frozenset([
            'Expansion.Conjunction','Comparison.Concession','Contingency.Cause','Temporal.Asynchronous','Temporal.Synchronous','Contingency.Condition','Comparison.Contrast','Expansion.Manner','Contingency.Purpose','Expansion.Instantiation','Expansion.Level-of-detail','Expansion.Substitution','Expansion.Disjunction','Contingency.Neg-condition'
            ])

    if set_name == "notconsidered_exp-L2-14way":
        return frozenset([
            'Comparison.Similarity','Contingency.Cond+SpeechAct','Contingency.Cause+Belief','Expansion.Exception','Expansion.Equivalence','Comparison.Conc+SpeechAct','Contingency.Cause+SpeechAct','Contingency.Negative-cause'
            ])
    
    if set_name == "considered_nexp-L2-14way+EntRel":
        return frozenset([
            'Expansion.Conjunction','Comparison.Concession','Contingency.Cause','Temporal.Asynchronous','Temporal.Synchronous','Contingency.Condition','Comparison.Contrast','Expansion.Manner','Contingency.Purpose','Expansion.Instantiation','Expansion.Level-of-detail','Expansion.Substitution','Contingency.Cause+Belief','Expansion.Equivalence', 'EntRel'
            ])

    if set_name == "notconsidered_nexp-L2-14way+EntRel":
        return frozenset([
            'Expansion.Disjunction','Contingency.Neg-condition','Comparison.Similarity','Contingency.Cond+SpeechAct','Expansion.Exception','Comparison.Conc+SpeechAct','Contingency.Cause+SpeechAct','Contingency.Negative-cause'
            ])
    
    if set_name == "considered_all+EntRel+NotMat":
        return frozenset([
            'Expansion.Conjunction','Comparison.Concession','Contingency.Cause','Temporal.Asynchronous','Temporal.Synchronous','Contingency.Condition','Comparison.Contrast','Expansion.Manner','Contingency.Purpose','Expansion.Instantiation','Expansion.Level-of-detail','Expansion.Substitution','Expansion.Disjunction','Contingency.Neg-condition','Comparison.Similarity','Contingency.Cond+SpeechAct','Contingency.Cause+Belief','Expansion.Exception','Expansion.Equivalence','Comparison.Conc+SpeechAct','Contingency.Cause+SpeechAct','Contingency.Negative-cause', 'NotMat', 'EntRel'
            ])
    
    if set_name == "notconsidered_all+EntRel+NotMat":
        return frozenset([
            ])

    if set_name == "considered_exp-L2-9way+NotCon":
        return frozenset([
            'Expansion.Conjunction','Comparison.Concession','Contingency.Cause','Temporal.Asynchronous','Temporal.Synchronous','Contingency.Condition','Comparison.Contrast','Expansion.Manner','Contingency.Purpose','NotCon'
            ])

    if set_name == "notconsidered_exp-L2-9way+NotCon":
        return frozenset([
            'Expansion.Instantiation','Expansion.Level-of-detail','Expansion.Substitution','Expansion.Disjunction','Contingency.Neg-condition','Comparison.Similarity','Contingency.Cond+SpeechAct','Contingency.Cause+Belief','Expansion.Exception','Expansion.Equivalence','Comparison.Conc+SpeechAct','Contingency.Cause+SpeechAct','Contingency.Negative-cause'
            ])
    
    if set_name == "considered_nexp-L2-9way+EntRel+NotCon":
        return frozenset([
            'Expansion.Conjunction','Comparison.Concession','Contingency.Cause','Temporal.Asynchronous','Comparison.Contrast','Expansion.Manner','Contingency.Purpose','Expansion.Instantiation','Expansion.Level-of-detail','EntRel','NotCon'
            ])

    if set_name == "notconsidered_nexp-L2-9way+EntRel+NotCon":
        return frozenset([
            'Temporal.Synchronous','Contingency.Condition','Expansion.Substitution','Expansion.Disjunction','Contingency.Neg-condition','Comparison.Similarity','Contingency.Cond+SpeechAct','Contingency.Cause+Belief','Expansion.Exception','Expansion.Equivalence','Comparison.Conc+SpeechAct','Contingency.Cause+SpeechAct','Contingency.Negative-cause'
            ])