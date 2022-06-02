import streamlit as st

reserved = ['all']

def initialize(d):
    for key in d.keys():
        if key in reserved:
            raise ValueError(f'"{key}" is a reserved key!')
        if key not in st.session_state:
            st.session_state[key] = d[key]


def getSession():
    return st.session_state


def setVar(name, val):
    if name in reserved:
        raise ValueError(f'"{name}" is a reserved key!')
    st.session_state[name] = val


def incVar(name):
    st.session_state[name] += 1


def getVar(name):
    return st.session_state[name]


def varExists(name):
    return name in st.session_state


def clear(name):
    if name == 'all':
        for key in st.session_state.keys():
            del st.session_state[key]
    else:
        del st.session_state[name]
