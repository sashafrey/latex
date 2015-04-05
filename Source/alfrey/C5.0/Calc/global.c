/*************************************************************************/
/*									 */
/*	 Source code for use with See5/C5.0 Release 2.09		 */
/*	 -----------------------------------------------		 */
/*		       Copyright RuleQuest Research 2012		 */
/*									 */
/*	This code is provided "as is" without warranty of any kind,	 */
/*	either express or implied.  All use is at your own risk.	 */
/*									 */
/*************************************************************************/

#include "defns.h"

/*************************************************************************/
/*									 */
/*		Parameters etc						 */
/*									 */
/*************************************************************************/

int		TRIALS=1,	/* number of trees to be grown */
		Trial;		/* trial number for boosting */

Boolean		RULES=0,	/* rule-based classifiers */
		RULESUSED=0;	/* list applicable rules */


/*************************************************************************/
/*									 */
/*		Attributes and data					 */
/*									 */
/*************************************************************************/

Attribute	ClassAtt=0,	/* attribute to use as class */
		LabelAtt,	/* attribute to use as case ID */
		CWtAtt;		/* attribute to use for case weight */

String		*ClassName=0,	/* class names */
		*AttName=0,	/* att names */
		**AttValName=0;	/* att value names */

char		*IgnoredVals=0;	/* values of labels and atts marked ignore */
int		IValsSize=0,	/* size of above */
		IValsOffset=0;	/* index of first free char */

int		MaxAtt,		/* max att number */
		MaxClass=0,	/* max class number */
		AttExIn=0,	/* attribute exclusions/inclusions */
		LineNo=0,	/* input line number */
		ErrMsgs=0,	/* errors found */
		Delimiter,	/* character at end of name */
		TSBase=0;	/* base day for time stamps */

DiscrValue	*MaxAttVal=0;	/* number of values for each att */

ContValue	*ClassThresh=0;	/* thresholded class attribute */

char		*SpecialStatus=0;/* special att treatment */

Definition	*AttDef=0;	/* definitions of implicit atts */

Boolean		*SomeMiss=Nil,	/* att has missing values */
		*SomeNA=Nil;	/* att has N/A values */

String		FileStem="undefined";

/*************************************************************************/
/*									 */
/*		Trees							 */
/*									 */
/*************************************************************************/

Tree		*Pruned=0;	/* decision trees */

ClassNo		*TrialPred=0;	/* predictions for each boost trial */

float		Confidence,	/* set by classify() */
		*Vote=0,	/* total votes for classes */
		**MCost=0;	/* misclass cost [pred][real] */

double		*ClassSum=0;	/* class weights during classification */

/*************************************************************************/
/*									 */
/*		Rules							 */
/*									 */
/*************************************************************************/

RuleNo		*Active=Nil,	/* rules that fire while classifying case */
		NActive,	/* number ditto */
		ActiveSpace=0,	/* space currently allocated for rules */
		*RulesUsed=Nil,	/* list of all rules used */
		NRulesUsed;	/* number ditto */

CRuleSet	*RuleSet=0;	/* rulesets */

ClassNo		Default;	/* default class associated with ruleset or
				   boosted classifier */

CRule		*MostSpec=0;	/* used in RuleClassify() */


/*************************************************************************/
/*									 */
/*		Misc							 */
/*									 */
/*************************************************************************/

FILE		*TRf=0;		/* file pointer for tree and rule i/o */
char		Fn[500];	/* file name */
