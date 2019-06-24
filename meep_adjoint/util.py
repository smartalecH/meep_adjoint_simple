"""General-purpose utilities for meep.adjoint

   This file collects some general-purpose utility functions and classes that are
   used in meep.adjoint but are not specific to adjoint solvers or computational
   electromagnetism.

   The functions/classes included are as follows:

   (1) process_options: creates {option:value} dicts by parsing a
                        hierarchy of config files, environment variables,
                        and command-line options

   (2) math utility routines

"""
from os import environ as env
from os.path import expanduser
import sys
import argparse
import configparser
from collections import namedtuple
from warnings import warn
from datetime import datetime as dt


OptionTemplate = namedtuple('OptionTemplate', 'name default help')
OptionTemplate.__doc__ = 'name, default value, and usage for one configurable option'


class OptionSettings(object):
    """Set option values from config files, command-line args, and environment

       On instantiation (and optional subsequent re-initialization), this class
       determines settings for a list of named options by considering each
       of the following sources, in increasing order of priority:

       (a) The default value given in the initialization template (see below)
       (b) Dict of customized defaults passed to constructor or set_defaults()
       (c) Global config file
       (d) Local config file
       (e) Environment variables
       (f) Command-line arguments. (Processed arguments are removed
           from sys.argv, leaving other arguments unchanged for subsequent
           downstream parsing.)

    Constructor arguments:

        templates (list of OptionTemplate): option definitions

        custom_defaults (dict): optional overrides of default option values

        section(str): optional section specification; if present, only matching
                      sections of config files are parsed. If absent, all
                      sections of all files are parsed.

        filename (str): If e.g. filename == 'myconfig.rc', then:
                          1. the global config file is ~/.my_config.rc
                          2. the local config file is my_config.rc in the current directory.
                        If unspecified, no configuration files are processed.

        search_env (bool): Whether or not to look for option settings in
                           environment variables. The default is True, but
                           you may want to set it to False if you have
                           option whose names overlap with standard
                           environment variables, such as 'HOME'
    """

    def __init__(self, templates, custom_defaults={},
                       section=None, filename=None, search_env=True):
        self.templates, self.section, self.filename = templates, section, filename
        self.process(templates, custom_defaults, section, filename)


    def set_defaults(self, custom_defaults={}):
        """re-process options with new default settings"""
        self.process(self.templates, custom_defaults, self.section, self.filename)


    def process(self, templates, custom_defaults, section, filename):
        """docstring goes here"""

        # initialize options to their template default values
        self.options  = { t.name: t.default for t in templates }

        # types inferred from defaults, used to type-check subsequent updates
        self.opttypes = { t.name: type(t.default) for t in templates }

        # update 1: caller-specified custom defaults
        self.revise(custom_defaults, 'custom_defaults')

        # updates 2,3: global, local config files
        if filename:
            fglob, floc = expanduser('~/.{}'.format(filename)), filename
            config = configparser.ConfigParser()
            config.read([fglob, floc])
            sections = config.sections()
            if section:
                sections=[s for s in sections if s.lower()==section.lower()]
            for s in sections:
                self.revise(config.items(s), 'config files')

        # update 4: environment variables
        envopts = { t.name: env[t.name] for t in templates if t.name in env }
        self.revise(envopts, 'environment variables')

        # update 5: command-line arguments
        parser = argparse.ArgumentParser()
        for n,v,h in [ (t.name, self.options[t.name], t.help) for t in templates ]:
            parser.add_argument('--{}'.format(n),type=type(v),default=v,help=h)
        argopts, leftovers = parser.parse_known_args()
        self.revise(argopts.__dict__.items(), 'command-line arguments')
        self.options['original_cmdline'] = ' '.join(sys.argv)
        sys.argv = [sys.argv[0]] + leftovers


    def revise(self, revisions, context):
        """cautiously apply a proposed set of updated option values

        Args:
            revisions: dict of {key,new_value} pairs OR list of (key,new_value) pairs
            context:   optional label like 'global config file' or 'command line'
                       for inclusion in error messages to help trace mishaps

        The result is similar to that of self.options.update( {k:v for (k,v) in revisions } ),
        but (a) ignores 'updates' to options that weren't previously configured
            (b) removes non-escaped single and double quotes surrounding strings,
            (c) attempts type conversions as necessary to preserve the type of the
                value associated with each key.

        Returns: none (vals is updated in place)
        """
        revisions = revisions.items() if hasattr(revisions,'items') else revisions
        for (key,newval) in [ (k,uq(v)) for k,v in revisions if k in self.options ]:
            try:
                self.options[key] = (self.opttypes[key])(newval)
            except ValueError:
                msg='option {}: ignoring improper value {} from {} (retaining value {})'
                warn(msg.format(key,newval,context,self.options[key]))


    def merge(self, partner):
        self.options.update(partner.options)


    def __call__(self, name, overrides={}):
        return overrides.get(name, self.options.get(name, None) )




def uq(s):
    """compensate for configparser's failure to unquote strings"""
    if s and isinstance(s,str) and s[0]==s[-1] and s[0] in ["'",'"']:
        return s[1:-1]
    return s



######################################################################
# Miscellaneous utility routines
######################################################################
def log(msg):
    from meep import am_master
    if not am_master(): return
    from meep.adjoint import options
    tm = dt.now().strftime("%T ")
    channel = options.get('logfile',None)
    if channel is not None:
        with open(channel,'a') as f:
            f.write("{} {}\n".format(tm,msg))
