package main;

import main.models.prior.BayesianNetwork;
import main.models.prior.NaiveMatrix;
import main.models.prior.PriorModel;
import main.models.sampling.AbstractSampler;
import main.models.sampling.AllSamples;
import main.models.sampling.RandomSamples;
import moa.recommender.rc.utils.Hash;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 3/05/16.
 **/
public class MarTVarD {
    private Instances dataSet1;
    private Instances dataSet2;
    private Instances dataSet;
    private AbstractSampler sampler;
    private String[][] orderedNple;
    private double[] distances;
    private double[] mean;
    private double[] sd;
    private double[][] localMax;
    private double[][] localMin;

    public MarTVarD(Instances dataSet1, Instances dataSet2) {
        this.dataSet1 = dataSet1;
        this.dataSet2 = dataSet2;
        this.dataSet = new Instances(this.dataSet1);
        this.dataSet.addAll(this.dataSet2);
        this.sampler = new RandomSamples(this.dataSet, 1000, 0);
    }

    /**
     * From the data set in the instance, order a the n-ple of the data sets variables
     * in order of largest to smallest Total Variation Distance
     * @param n The number of variables in a n-ple, (2 = tuple, 3 = triple, and so on)
     * @return An ordered list of n-ples
     */
    public String[][] findOrderedNPle(int n) {
        int[] elements = new int[dataSet1.numAttributes() - 1];
        for (int i = 0; i < dataSet1.numAttributes() - 1; i++) elements[i] = i;

        int nCombination = nCr(dataSet1.numAttributes() - 1, n);
        Map<int[], Double> sets = new HashMap<>();
        Map<int[], Double> means = new HashMap<>();
        Map<int[], Double> sds = new HashMap<>();
        Map<int[], double[]> maximums = new HashMap<>();
        Map<int[], double[]> minimums = new HashMap<>();
        for (int i = 0; i < nCombination; i++) {
            int[] indices = getKthCombination(i, elements, n);
            Instances instances1 = seperateVariables(this.dataSet1, indices);
            NaiveMatrix model1 = new NaiveMatrix(instances1);
            Instances instances2 = seperateVariables(this.dataSet2, indices);
            NaiveMatrix model2 = new NaiveMatrix(instances2);
            Instances allInstances = AbstractSampler.findIntersectionBetweenInstances(instances1, instances2);
            // TODO: Find suitable sample size
            //AbstractSampler sampler = new RandomSamples(allInstances, allInstances.size() / 1000, 0);
            AbstractSampler sampler = new AllSamples(allInstances);

            double distance = model1.findDistance(model1, model2, sampler) * sampler.getMagnitudeScale();
            sets.put(indices, distance);
            means.put(indices, model1.getMean());
            sds.put(indices, model1.getSd());
            maximums.put(indices, model1.getMax());
            minimums.put(indices, model1.getMin());
        }

        sets = sortByValue(sets);
        this.distances = new double[nCombination];
        this.mean = new double[nCombination];
        this.sd = new double[nCombination];
        this.localMin = new double[nCombination][n+1];
        this.localMax = new double[nCombination][n+1];
        this.orderedNple = new String[nCombination][n];

        int[][] orderedSets;
        orderedSets = sets.keySet().toArray(new int[nCombination][n]);
        for (int i = 0; i < nCombination; i++) {
            this.distances[i] = sets.get(orderedSets[i]);
            this.mean[i] = means.get(orderedSets[i]);
            this.sd[i] = sds.get(orderedSets[i]);
            this.localMax[i] = maximums.get(orderedSets[i]);
            this.localMin[i] = minimums.get(orderedSets[i]);
            for (int j = 0; j < orderedSets[i].length; j++) {
                this.orderedNple[i][j] = dataSet1.attribute(orderedSets[i][j]).name();
            }
        }
        return this.generateOutput();
    }

    private String[][] generateOutput() {
        String[][] results = new String[this.orderedNple.length][6];
        for (int i = 0; i < this.orderedNple.length; i++) {
            results[i][0] = Double.toString(this.distances[i]);
            results[i][1] = Double.toString(this.mean[i]);
            results[i][2] = Double.toString(this.sd[i]);
            results[i][3] = Double.toString(this.localMax[i][0]);
            results[i][4] = Double.toString(this.localMin[i][0]);
            results[i][5] = "";
            for (int j = 0; j < this.orderedNple[i].length; j++) {
                results[i][3] += "_" + this.orderedNple[i][j] + "=" + this.localMax[i][1+j];
                results[i][4] += "_" + this.orderedNple[i][j] + "=" + this.localMin[i][1+j];
                results[i][5] += "_" + this.orderedNple[i][j];
            }
        }
        return results;
    }

    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue( Map<K, V> map ) {
        List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>( map.entrySet() );
        Collections.sort( list, new Comparator<Map.Entry<K, V>>() {
            public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 )
            {
                return (o1.getValue()).compareTo( o2.getValue() );
            }
        } );

        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list)
        {
            result.put( entry.getKey(), entry.getValue() );
        }
        return result;
    }

    private int[] getKthCombination(int k, int[] elements, int choices) {
        if (choices == 0) return new int[]{};
        else if (elements.length == choices) return  elements;
        else {
            int nCombinations = nCr(elements.length - 1, choices - 1);
            if (k < nCombinations) return ArrayUtils.addAll(ArrayUtils.subarray(elements, 0, 1),
                    getKthCombination(k, ArrayUtils.subarray(elements, 1, elements.length), choices - 1));
            else return getKthCombination(k - nCombinations, ArrayUtils.subarray(elements, 1, elements.length), choices);
        }
    }

    private int nCr(int n, int r) {
        if (r >= n /2) r = n - r;
        int ans = 1;
        for (int i = 1; i <= r; i++) {
            ans *= n - r + i;
            ans /= i;
        }
        return ans;
    }

    private int factorial(int n) {
        if (n <= 0) return 1;
        else {
            return (n * factorial(n-1));
        }
    }


    private Instances seperateVariables(Instances instances, int[] variableIndices) {
        ArrayList<ArrayList<String>> allValues = this.sampler.getAllPossibleValues();
        // Create new Instances
        String name = "";
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int index : variableIndices) {
            name += "v"+index;
            attributes.add(new Attribute("v"+index, allValues.get(index), instances.attribute(index).getMetadata()));
        }
        attributes.add(new Attribute("class", instances.attribute(instances.classIndex()).getMetadata()));
        Instances newInstances = new Instances(name, attributes, instances.size());
        newInstances.setClassIndex(variableIndices.length);

        // Copy value of certain variables from instances to new instances
        for (Instance instance : instances) {
            double[] inst = new double[variableIndices.length + 1];
            for (int i = 0; i < variableIndices.length; i++) {
                inst[i] = instance.value(variableIndices[i]);
            }
            Instance newInstance = new DenseInstance(1, inst);
            newInstance.setDataset(newInstances);
            newInstance.setClassMissing();
            newInstances.add(newInstance);
        }

        return newInstances;
    }
}
