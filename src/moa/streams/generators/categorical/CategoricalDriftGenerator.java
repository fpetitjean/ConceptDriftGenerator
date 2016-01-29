/*
 * Created by Lee Loong Kuan on 5/09/2015.
 * Interface that represents a basic Concept Drift Generator
 * Contains static methods to generate probabilities
 */

package moa.streams.generators.categorical;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;

import moa.core.FastVector;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class CategoricalDriftGenerator extends AbstractOptionHandler implements InstanceStream {

    private static final long serialVersionUID = -3582081197584644644L;
    
    public IntOption nAttributes = new IntOption("nAttributes", 'n',
	    "Number of attributes as parents of the class", 2, 1, 10);
    public IntOption nValuesPerAttribute = new IntOption("nValuesPerAttribute", 'v',
	    "Number of values per attribute", 2, 2, 5);
    public IntOption burnInNInstances = new IntOption("burnInNInstances", 'b',
	    "Number of instances before the start of the drift", 10000, 0, Integer.MAX_VALUE);
    public FloatOption driftMagnitudePrior = new FloatOption("driftMagnitudePrior", 'i',
	    "Magnitude of the drift between the starting probability and the one after the drift."
		    + " Magnitude is expressed as the Hellinger distance [0,1]", 0.5, 1e-20, 0.9);
    public FloatOption driftMagnitudeConditional = new FloatOption("driftMagnitudeConditional",
	    'o',
	    "Magnitude of the drift between the starting probability and the one after the drift."
		    + " Magnitude is expressed as the Hellinger distance [0,1]", 0.5, 1e-20, 0.9);
    public FloatOption precisionDriftMagnitude = new FloatOption(
	    "epsilon",
	    'e',
	    "Precision of the drift magnitude for p(x) (how far from the set magnitude is acceptable)",
	    0.01, 1e-20, 1.0);
    public FlagOption driftConditional = new FlagOption("driftConditional", 'c',
	    "States if the drift should apply to the conditional distribution p(y|x).");
    public FlagOption driftPriors = new FlagOption("driftPriors", 'p',
	    "States if the drift should apply to the prior distribution p(x). ");
    public IntOption seed = new IntOption("seed", 'r', "Seed for random number generator", -1,
	    Integer.MIN_VALUE, Integer.MAX_VALUE);

    FastVector<Attribute> getHeaderAttributes(int nAttributes, int nValuesPerAttribute) {

	FastVector<Attribute> attributes = new FastVector<>();
	List<String> attributeValues = new ArrayList<String>();
	for (int v = 0; v < nValuesPerAttribute; v++) {
	    attributeValues.add("v" + (v + 1));
	}
	for (int i = 0; i < nAttributes; i++) {
	    attributes.addElement(new Attribute("x" + (i + 1), attributeValues));
	}
	List<String> classValues = new ArrayList<String>();

	for (int v = 0; v < nValuesPerAttribute; v++) {
	    classValues.add("class" + (v + 1));
	}
	attributes.addElement(new Attribute("class", classValues));

	return attributes;
    }

    public static void generateRandomPyGivenX(double[][] pygx, RandomDataGenerator r) {
	for (int i = 0; i < pygx.length; i++) {
	    double[] lineCPT = pygx[i];
	    int chosenClass = r.nextSecureInt(0, lineCPT.length - 1);

	    for (int c = 0; c < lineCPT.length; c++) {
		if (c == chosenClass) {
		    lineCPT[c] = 1.0;
		} else {
		    lineCPT[c] = 0.0;
		}
	    }
	}

    }
    
    public static void generateRandomPx(double[][] px, RandomDataGenerator r) {
	generateRandomPx(px, r,false);
    }

    public static void generateRandomPx(double[][] px, RandomDataGenerator r,boolean verbose) {
	double sum;
	for (int a = 0; a < px.length; a++) {
	    sum = 0.0;
	    for (int v = 0; v < px[a].length; v++) {
		px[a][v] = r.nextGamma(1.0,1.0);
		sum += px[a][v];
	    }
	    // normalizing
	    for (int v = 0; v < px[a].length; v++) {
		px[a][v] /= sum;
		 
	    }
	    if(verbose)System.out.println("p(x_" + a + ")=" + Arrays.toString(px[a]));
	}

    }
    
    public static void generateUniformPx(double[][] px) {
	for (int a = 0; a < px.length; a++) {
	    for (int v = 0; v < px[a].length; v++) {
		px[a][v] = 1.0/px[a].length;
	    }
	}

    }

    public static double computeMagnitudePX(int nbCombinationsOfValuesPX, double[][] base_px,
	    double[][] drift_px) {
	int[] indexes = new int[base_px.length];
	double m = 0.0;
	for (int i = 0; i < nbCombinationsOfValuesPX; i++) {
	    getIndexes(i, indexes, base_px[0].length);
	    double p = 1.0, q = 1.0;
	    for (int a = 0; a < indexes.length; a++) {
		p *= base_px[a][indexes[a]];
		q *= drift_px[a][indexes[a]];
	    }
	    double diff = Math.sqrt(p) - Math.sqrt(q);
	    m += diff * diff;
	}
	m = Math.sqrt(m) / Math.sqrt(2);
	return m;
    }

    public static double computeMagnitudePYGX(double[][] base_pygx, double[][] drift_pygx) {
	double magnitude = 0.0;
	for (int i = 0; i < base_pygx.length; i++) {
	    double partialM = 0.0;
	    for (int c = 0; c < base_pygx[i].length; c++) {
		double diff = Math.sqrt(base_pygx[i][c]) - Math.sqrt(drift_pygx[i][c]);
		partialM += diff * diff;
	    }
	    partialM = Math.sqrt(partialM) / Math.sqrt(2);
	    assert (partialM == 0.0 || partialM == 1.0);
	    magnitude += partialM;
	}
	magnitude /= base_pygx.length;
	return magnitude;
    }
    
    public static double computeMagnitudeClassPrior(double[] baseClassP, double[] driftClassP) {
	    double magnitude = 0.0;
	    for (int c = 0; c < baseClassP.length; c++) {
		double diff = Math.sqrt(baseClassP[c]) - Math.sqrt(driftClassP[c]);
		magnitude += diff * diff;
	    }
	    magnitude = Math.sqrt(magnitude) / Math.sqrt(2);
	return magnitude;
    }


    static void getIndexes(int index, int[] indexes, int nValuesPerAttribute) {
	for (int i = indexes.length - 1; i > 0; i--) {
	    int dim = nValuesPerAttribute;
	    indexes[i] = index % dim;
	    index /= dim;
	}
	indexes[0] = index;
    }

    public static double[] computeClassPrior(double[][] px, double[][] pygx) {
	int nClasses = pygx[0].length;
	int nAttributes = px.length;
	int nValuesPerAttribute = px[0].length;
	double []classPrior = new double[nClasses];
	int[] indexes = new int[nAttributes];
	for (int lineCPT = 0; lineCPT < pygx.length; lineCPT++) {
	    getIndexes(lineCPT, indexes, nValuesPerAttribute);
	    double probaLine = 1.0;
	    for (int a = 0; a < indexes.length; a++) {
		probaLine *= px[a][indexes[a]];
	    }
	    for (int c = 0; c < nClasses; c++) {
		classPrior[c]+=probaLine*pygx[lineCPT][c];
	    }
	}
	return classPrior;
	
    }

}
