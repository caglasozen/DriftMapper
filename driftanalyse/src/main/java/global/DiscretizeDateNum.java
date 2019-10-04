package global;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.*;

/*
 * Created by LoongKuan on 3/07/2017.
 */

public class DiscretizeDateNum {
    private Map<Integer, List<String>> allDates;
    private Map<Integer, double[]> allCutPoints;

    public Instances getDiscreteStructure() {
        return discreteStructure;
    }

    private Instances discreteStructure;

    public DiscretizeDateNum(ConverterUtils.DataSource dataSource, Instances instanceInfo) {
        try {
            ArrayList<Integer> dateIndices = new ArrayList<>();
            ArrayList<Integer> numericIndices = new ArrayList<>();
            for (int i = 0; i < instanceInfo.numAttributes(); i++) {
                if (instanceInfo.attribute(i).isDate()) dateIndices.add(i);
                else if (instanceInfo.attribute(i).isNumeric()) numericIndices.add(i);
            }

            Instances sampledInstances = sampleInstances(1000, dataSource, instanceInfo);
            this.allDates = getAllDates(
                    dataSource,
                    dateIndices.stream().mapToInt(Integer::intValue).toArray());
            this.allCutPoints = getAllCutPoints(
                    sampledInstances,
                    numericIndices.stream().mapToInt(Integer::intValue).toArray());
            this.discreteStructure = generateDiscreteStructure(instanceInfo);
            this.discreteStructure.setClassIndex(instanceInfo.classIndex());
            dataSource.reset();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public Instance discretizeInstance(Instance instance) {
        double[] attValues = new double[instance.numAttributes()];
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.attribute(i).isDate()) {
                attValues[i] = allDates.get(i).indexOf(instance.toString(i));
            }
            else if (instance.attribute(i).isNumeric()) {
                double val = instance.value(i);
                double[] cutOff = this.allCutPoints.get(i);
                if (val <= cutOff[0]) attValues[i] = 0.0;
                for (int j = 1; j < cutOff.length; j++) {
                    if (val > cutOff[j-1] && val <= cutOff[j]) attValues[i] = (double)j;
                }
                if (val > cutOff[cutOff.length - 1]) attValues[i] = (double)cutOff.length;
            }
            else {
                attValues[i] = instance.value(i);
            }
        }
        Instance newInstance = new DenseInstance(1.0, attValues);
        newInstance.setDataset(this.discreteStructure);
        return newInstance;
    }

    private Instances generateDiscreteStructure(Instances instanceInfo) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < instanceInfo.numAttributes(); i++) {
            Attribute attribute;
            if (instanceInfo.attribute(i).isDate()) {
                attribute = new Attribute(
                        instanceInfo.attribute(i).name(),
                        this.allDates.get(i));
            }
            else if (instanceInfo.attribute(i).isNumeric()) {
                attribute = new Attribute(
                        instanceInfo.attribute(i).name(),
                        generateDiscreteValues(this.allCutPoints.get(i)));
            }
            else {
                attribute = instanceInfo.attribute(i);
            }
            attributes.add(attribute);
        }
        return new Instances(instanceInfo.relationName(), attributes, 0);
    }

    private static List<String> generateDiscreteValues(double[] cutPoints) {
        List<String> values = new ArrayList<>();
        values.add("(-Inf-" + Double.toString(cutPoints[0]) + "]");
        for (int i = 0; i < cutPoints.length - 1; i++) {
            values.add("(" + cutPoints[i] + "-" + cutPoints[i+1] + "]");
        }
        values.add("(" + cutPoints[cutPoints.length - 1] + "-Inf)");
        return values;
    }

    private static Instances sampleInstances(int nSamples, ConverterUtils.DataSource dataSource, Instances structure) throws Exception{
        Instances instances = new Instances(structure, nSamples);
        Random random = new Random();
        dataSource.reset();
        int currentInstance = 0;
        // Iterate through all instances in file
        while (dataSource.hasMoreElements(structure)) {
            currentInstance++;
            Instance instance = dataSource.nextElement(structure);
            // Add first nSamples instance to samples
            if (currentInstance <= nSamples) {
                instances.add(instance);
            }
            // for instance at t > nSamples, it has nSamples/t probability to replace a random sample
            else {
                int rand = random.nextInt(currentInstance);
                if (rand < nSamples) {
                    instances.delete(rand);
                    instances.add(instance);
                }
            }
        }
        return instances;
    }

    private static Map<Integer, List<String>> getAllDates(ConverterUtils.DataSource dataSource,
                                                          int[] attributeIndices) throws Exception{
        dataSource.reset();
        Map<Integer, List<String>> dateMap = new HashMap<>();
        Map<Integer, String> previousDate = new HashMap<>();
        Instances structure = dataSource.getStructure();
        for (int i : attributeIndices) {
            assert structure.attribute(i).isDate();
        }
        dataSource.reset();
        while (dataSource.hasMoreElements(structure)) {
            Instance instance = dataSource.nextElement(structure);
            for (int index :attributeIndices) {
                String currentDate = instance.toString(index);
                if (!dateMap.containsKey(index)) {
                    dateMap.put(index, new ArrayList<>());
                    dateMap.get(index).add(currentDate);
                    previousDate.put(index, currentDate);
                }
                if (!currentDate.equals(previousDate.get(index))) {
                    dateMap.get(index).add(currentDate);
                    previousDate.put(index, currentDate);
                }
            }
        }
        return dateMap;
    }

    private static Map<Integer, double[]> getAllCutPoints(Instances sampledInstances,
                                                          int[] attributeIndices) throws Exception {
        Discretize filter = new Discretize();
        filter.setUseEqualFrequency(true);
        filter.setBins(5);
        filter.setAttributeIndicesArray(attributeIndices);
        filter.setInputFormat(sampledInstances);
        Filter.useFilter(sampledInstances, filter);
        Map<Integer, double[]> allCutPoints = new HashMap<>();
        for (int index : attributeIndices) {
            allCutPoints.put(index, filter.getCutPoints(index));
        }
        return allCutPoints;
    }
}
