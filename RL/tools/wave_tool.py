import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
import spec_tools.spec_tools as st


def peakdet(v, delta, x=None):
	"""
	Converted from MATLAB script at http://billauer.co.il/peakdet.html

	Returns two arrays

	function [maxtab, mintab]=peakdet(v, delta, x)
	%PEAKDET Detect peaks in a vector
	%        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
	%        maxima and minima ("peaks") in the vector V.
	%        MAXTAB and MINTAB consists of two columns. Column 1
	%        contains indices in V, and column 2 the found values.
	%
	%        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
	%        in MAXTAB and MINTAB are replaced with the corresponding
	%        X-values.
	%
	%        A point is considered a maximum peak if it has the maximal
	%        value, and was preceded (to the left) by a value lower by
	%        DELTA.

	% Eli Billauer, 3.4.05 (Explicitly not copyrighted).
	% This function is released to the public domain; Any use is allowed.

	"""
	maxtab = []
	mintab = []

	if x is None:
		x = arange(len(v))

	v = asarray(v)

	if len(v) != len(x):
		sys.exit('Input vectors v and x must have same length')

	if not isscalar(delta):
		sys.exit('Input argument delta must be a scalar')

	if delta <= 0:
		sys.exit('Input argument delta must be positive')

	mn, mx = Inf, -Inf
	mnpos, mxpos = NaN, NaN

	lookformax = True

	for i in arange(len(v)):
		this = v[i]
		if this > mx:
			mx = this
			mxpos = x[i]
		if this < mn:
			mn = this
			mnpos = x[i]

		if lookformax:
			if this < mx - delta:
				maxtab.append((mxpos, mx))
				mn = this
				mnpos = x[i]
				lookformax = False
		else:
			if this > mn + delta:
				mintab.append((mnpos, mn))
				mx = this
				mxpos = x[i]
				lookformax = True

	return array(maxtab), array(mintab)


def return_wave_heights(waves):
	all_crest, all_trough = peakdet(waves, 0.1)
	return all_crest[0:len(all_trough), 1] - all_trough[:, 1]

if __name__=="__main__":
	from matplotlib.pyplot import plot, scatter, show
	# series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]

	wave = st.Spectrum.from_synthetic(spreading=None, Hs=1, Tp=8)
	relative_RAO = st.Rao.from_liftdyn('../RAO/relative_motion.stf_plt', 1, 1)
	resp = relative_RAO.get_response(wave)
	for i in range(10000):
		rel_motion_sc_t, rel_motion_sc = resp.make_time_trace(2000, 0.2)
		series = rel_motion_sc
		maxtab, mintab = peakdet(series,0.1)
		if len(maxtab) - len(mintab) == 0:
			height = maxtab[:, 1]-mintab[:, 1]
			scatter(array(mintab)[:,0], height)
			plot(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
			plot(array(mintab)[:,0], array(mintab)[:,1], color='red')
			plot(series)
			show()