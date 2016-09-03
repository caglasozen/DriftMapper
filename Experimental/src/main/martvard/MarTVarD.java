package main.martvard;

import main.models.NaiveMatrix;
import main.models.sampling.AbstractSampler;
import main.models.sampling.AllSamples;
import main.models.sampling.RandomSamples;
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

    private Map<int[], double[]> means;
    private Map<int[], double[]> sds;
    private Map<int[], double[][]> maximums;
    private Map<int[], double[][]> minimums;
    private ArrayList<Map<int[], Double>> setDistMap;
    private int nAttributes;

    /**
     * From the data set in the instance, order a the n-ple of the data sets variables
     * in order of largest to smallest Total Variation Distance
     * @param n The number of variables in a n-ple, (2 = tuple, 3 = triple, and so on)
     * @return An ordered list of n-ples
     */
    public String[][][] findOrderedNPle(int n) {
        nAttributes = n;
        means = new HashMap<>();
        sds = new HashMap<>();
        maximums = new HashMap<>();
        minimums = new HashMap<>();
        setDistMap = new ArrayList<>();

        int[] attributeIndices = new int[dataSet1.numAttributes() - 1];
        for (int i = 0; i < dataSet1.numAttributes() - 1; i++) attributeIndices[i] = i;

        int nCombination = nCr(dataSet1.numAttributes() - 1, n);
        for (int i = 0; i < 4; i++) {
            setDistMap.add(new HashMap<>());
        }
        for (int i = 0; i < nCombination; i++) {
            // Get combination between attributes
            int[] indices = getKthCombination(i, attributeIndices, n);
            Instances instances1 = seperateVariables(this.dataSet1, indices);
            NaiveMatrix model1 = new NaiveMatrix(instances1);
            Instances instances2 = seperateVariables(this.dataSet2, indices);
            NaiveMatrix model2 = new NaiveMatrix(instances2);
            Instances allInstances = AbstractSampler.findUnionBetweenInstances(instances1, instances2);
            // TODO: Find suitable sample size
            //AbstractSampler sampler = new RandomSamples(allInstances, allInstances.size() / 1000, 0);
            AbstractSampler sampler = new AllSamples(allInstances);

            // p(v), p(v|y), p(y), p(y|v)
            double[] distances = model1.findDistance(model1, model2, sampler);

            for (int j = 0; j < distances.length; j++) {
                setDistMap.get(j).put(indices, distances[j]);
            }
            means.put(indices, model1.getMean().clone());
            sds.put(indices, model1.getSd().clone());
            maximums.put(indices, model1.getMax().clone());
            minimums.put(indices, model1.getMin().clone());
        }

        for (int i = 0; i < 4; i++) {
            setDistMap.set(i, sortByValue(setDistMap.get(i)));
        }
        this.distances = new double[nCombination];
        this.mean = new double[nCombination];
        this.sd = new double[nCombination];
        // min and max need extra 2 for both distance and class
        this.localMin = new double[nCombination][n+2];
        this.localMax = new double[nCombination][n+2];
        // need extra 1 space for class
        this.orderedNple = new String[nCombination][n+1];

        int[][] orderedSets;
        String[][][] outs = new String[4][this.orderedNple.length][8];
        for (int i = 0; i < 4; i++) {
            orderedSets = setDistMap.get(i).keySet().toArray(new int[nCombination][n]);
            for (int j = 0; j < nCombination; j++) {
                this.distances[j] = setDistMap.get(i).get(orderedSets[j]);
                this.mean[j] = means.get(orderedSets[j])[i];
                this.sd[j] = sds.get(orderedSets[j])[i];
                this.localMax[j] = maximums.get(orderedSets[j])[i];
                this.localMin[j] = minimums.get(orderedSets[j])[i];
                for (int k = 0; k < orderedSets[j].length; k++) {
                    this.orderedNple[j][k] = dataSet1.attribute(orderedSets[j][k]).name();
                }
                this.orderedNple[j][orderedSets[j].length] = dataSet1.attribute(dataSet1.numAttributes() - 1).name();
            }
            outs[i] = this.generateOutput();
        }
        return outs;
    }

    public String[][] getClassSummary() {
        int nCombination = setDistMap.get(2).size();
        int[][] attributeIndices = setDistMap.get(2).keySet().toArray(new int[nCombination][nAttributes]);

        double[] distances = new double[nCombination];
        double[] mean = new double[nCombination];
        double[] sd = new double[nCombination];
        // min and max need just 2 for both distance and class
        double[][] localMin = new double[nCombination][2];
        double[][] localMax = new double[nCombination][2];
        // need extra 1 space for class
        String[][] orderedNple = new String[nCombination][nAttributes+1];

        // Extract relevant data
        for (int j = 0; j < nCombination; j++) {
            distances[j] = setDistMap.get(2).get(attributeIndices[j]);
            mean[j] = means.get(attributeIndices[j])[2];
            sd[j] = sds.get(attributeIndices[j])[2];
            localMax[j] = maximums.get(attributeIndices[j])[2];
            localMin[j] = minimums.get(attributeIndices[j])[2];
            for (int k = 0; k < attributeIndices[j].length; k++) {
                orderedNple[j][k] = dataSet1.attribute(attributeIndices[j][k]).name();
            }
            orderedNple[j][attributeIndices[j].length] = dataSet1.attribute(dataSet1.numAttributes() - 1).name();
        }

        // Generate output
        String[][] results = new String[nCombination][8];
        return new String[][]{};
    }

    private String[][] generateOutput() {
        String[][] results = new String[this.orderedNple.length][8];
        for (int i = 0; i < this.orderedNple.length; i++) {
            results[i][0] = Double.toString(this.distances[i]);
            results[i][1] = Double.toString(this.mean[i]);
            results[i][2] = Double.toString(this.sd[i]);
            results[i][3] = Double.toString(this.localMax[i][0]);
            results[i][4] = "";
            results[i][5] = Double.toString(this.localMin[i][0]);
            results[i][6] = "";
            results[i][7] = "";
            for (int j = 0; j < this.orderedNple[i].length; j++) {
                String minVal = (int)this.localMin[i][1 + j] < 0 ? "*" :
                        dataSet.attribute(this.orderedNple[i][j]).value((int)this.localMin[i][1 + j]);
                String maxVal = (int)this.localMax[i][1 + j] < 0 ? "*" :
                        dataSet.attribute(this.orderedNple[i][j]).value((int)this.localMax[i][1 + j]);
                results[i][4] += "_" + this.orderedNple[i][j] + "=" + maxVal;
                results[i][6] += "_" + this.orderedNple[i][j] + "=" + minVal;
                if (j < this.orderedNple[i].length - 1) results[i][7] += "_" + this.orderedNple[i][j];
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
            inst[inst.length - 1] = instance.classValue();
            Instance newInstance = new DenseInstance(1, inst);
            newInstance.setDataset(newInstances);
            newInstances.add(newInstance);
        }

        return newInstances;
    }
}
