package main;

import main.models.prior.BayesianNetwork;
import main.models.prior.PriorModel;
import main.models.sampling.AbstractSampler;
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
    public int[][] findOrderedNPle(int n) {
        int[] elements = new int[dataSet1.numAttributes() - 1];
        for (int i = 0; i < dataSet1.numAttributes() - 1; i++) elements[i] = i;

        int nCombination = nCr(dataSet1.numAttributes() - 1, n);
        Map<int[], Double> sets = new HashMap<>();
        for (int i = 0; i < nCombination; i++) {
            int[] indices = getKthCombination(i, elements, n);
            Instances instances1 = seperateVariables(this.dataSet1, indices);
            PriorModel model1 = new BayesianNetwork(instances1);
            Instances instances2 = seperateVariables(this.dataSet2, indices);
            PriorModel model2 = new BayesianNetwork(instances2);
            Instances allInstances = AbstractSampler.findIntersectionBetweenInstances(instances1, instances2);
            AbstractSampler sampler = new RandomSamples(allInstances, 1000, 0);

            double distance = model1.findDistance(model1, model2, sampler);
            sets.put(indices, distance);
        }

        sets = sortByValue(sets);

        int[][] orderedSets;
        orderedSets = sets.keySet().toArray(new int[nCombination][n]);
        return orderedSets;
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
            int nCombinations = nCr(elements.length - 1, choices);
            if (k < nCombinations) return getKthCombination(k, ArrayUtils.subarray(elements, 1, elements.length), choices);
            else return ArrayUtils.addAll(ArrayUtils.subarray(elements, 0, 1),
                    getKthCombination(k - nCombinations, ArrayUtils.subarray(elements, 1, elements.length), choices - 1));
        }
    }

    private int nCr(int n, int r) {
        return factorial(n) / (factorial(r) * factorial(n-r));
    }

    private int factorial(int n) {
        if (n == 0) return 1;
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
