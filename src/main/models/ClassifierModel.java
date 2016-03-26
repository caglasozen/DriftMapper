package main.models;

import main.generator.componets.BayesianNetworkGenerator;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import explorer.ChordalysisModelling;
import main.models.distance.Distance;
import main.models.distance.HellingerDistance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.DoubleStream;

/**
 * Created by Lee on 30/01/2016.
 **/
public abstract class ClassifierModel {
    SamoaToWekaInstanceConverter samoaConverter = new SamoaToWekaInstanceConverter();
    WekaToSamoaInstanceConverter wekaConverter = new WekaToSamoaInstanceConverter();

    // List of Dictionaries that map a possible value for a X variable to it's frequency
    // Each dictionary represents the frequency distribution for an attribute
    protected ArrayList<ArrayList<String>> xPossibleValues = new ArrayList<>();

    Instances allPossibleInstances;
    protected HashMap<Integer, Instance> hashedInstanceSet;
    public Instances dataSet;

    protected BayesianNetworkGenerator bnModel;

    protected void resetModel() {
        trainClassifier();
        generateBayesNet();
    }

    public void changeDataSet(Instances dataSet) {
        this.dataSet = dataSet;
        resetModel();
    }

    abstract void trainClassifier();

    public abstract ClassifierModel copy();

    protected void generateBayesNet() {
        String[] variablesNames = new String[dataSet.numAttributes()];
        for (int i = 0; i < variablesNames.length; i++) {
            variablesNames[i] = dataSet.attribute(i).name();
        }

        Instances trimmedInstances = trimClass(dataSet);
        // Chordalysis modeler with 0.1G of memory allocated
        ChordalysisModelling modeller = new ChordalysisModelling(0.05);
        modeller.buildModel(new Instances(trimmedInstances));

        bnModel = new BayesianNetworkGenerator(modeller, variablesNames, xPossibleValues);
    }

    protected static Instances trimClass(Instances instances) {
        // Get attributes
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < instances.numAttributes(); i++) {
            // if the current attribute is not a class attribute, add the attribute
            if (!(instances.classIndex() == i)) attributes.add(instances.attribute(i));
        }
        Instances instancesReturn = new Instances("Trimmed Data", attributes, instances.size());
        instancesReturn.setClassIndex(instancesReturn.numAttributes()-1);

        for (int i = 0; i < instances.size(); i++) {
            DenseInstance instance = new DenseInstance(instances.get(i));
            instance.deleteAttributeAt(instances.classIndex());
            instance.setDataset(instancesReturn);
            instancesReturn.add(instance);
        }
        return instancesReturn;
    }

    protected void getXMeta() {
        ArrayList<Map<String, Integer>> xValueToFreq = new ArrayList<>();
        // NOTE: j represents x_n
        for (int j = 0; j < dataSet.numAttributes() - 1; j++) {
            // Initialise local variables for this loop (aka. data for this X variable)
            Map<String, Integer> curDict = new HashMap<>();
            for (int i = 0; i < dataSet.size(); i++) {
                String curKey = Double.toString(dataSet.instance(i).value(j));
                int curFreq = curDict.containsKey(curKey) ? curDict.get(curKey) + 1 : 1;
                curDict.put(curKey, curFreq);
            }
            // Add completed Map/Dict to the List
            xValueToFreq.add(curDict);
            // Add all the keys/Possible values to xPossibleValues
            xPossibleValues.add(new ArrayList<>(curDict.keySet()));
        }

        //Suggest Garbage collector to run
        System.gc();

        int nCombinations = 1;
        for (ArrayList<String> values : xPossibleValues) {
            nCombinations *= values.size();
        }
        allPossibleInstances = new Instances(dataSet, nCombinations);

        // Generate set of all appeared combinations of x values

        generateCombinations(0, new ArrayList<>());
        //Suggest Garbage collector to run
        System.gc();
    }

    public abstract double findPygv(int classLabel, Instance xValVector);

    /**
     * Calculate the probability of a certain combination of x values (aka. xVector) occurring
     * @param xValVector The combination of x values (xVector)
     * @return Probability of given vector of x values occurring
     */
    public abstract double findPv(double[] xValVector);

    protected static double[] getNormalisedVotes(double[] unNormalisedVotes, int numberNormVotes) {
        double voteSum = DoubleStream.of(unNormalisedVotes).sum();
        double[] normalisedVotes = new double[numberNormVotes];
        for (int i = 0; i < unNormalisedVotes.length; i++) {
            normalisedVotes[i] = (voteSum == 0.0 || Double.isNaN(voteSum)) ? 0.0f : unNormalisedVotes[i] / voteSum;
        }
        return normalisedVotes;
    }

    public static double pygvModelDistance(EnsembleClassifierModel modelBD, EnsembleClassifierModel modelAD) {
        WekaToSamoaInstanceConverter converter = new WekaToSamoaInstanceConverter();

        Instances allPossibleInstances = findIntersectionBetweenInstances(modelBD.allPossibleInstances, modelAD.allPossibleInstances);

        double[] totalDist = new double[modelBD.baseClassifiers.length];
        for (int k = 0; k < modelBD.baseClassifiers.length; k++) {
            totalDist[k] = 0.0f;
            for (int j = 0; j < allPossibleInstances.size(); j++) {
                Instance inst = allPossibleInstances.get(j);
                double driftDist = 0.0f;
                double[] pygvBD = modelBD.baseClassifiers[k].getVotesForInstance(converter.samoaInstance(inst));
                double[] pygvAD = modelAD.baseClassifiers[k].getVotesForInstance(converter.samoaInstance(inst));
                // Normalise votes
                int numClasses = (pygvAD.length < pygvBD.length) ? pygvBD.length : pygvAD.length;
                double[] pygvBDNorm = getNormalisedVotes(pygvBD, numClasses);
                double[] pygvADNorm = getNormalisedVotes(pygvAD, numClasses);
                // Get Distance
                for (int i = 0; i < numClasses; i++) {
                    driftDist = driftDist + Math.pow((Math.sqrt(pygvBDNorm[i]) - Math.sqrt(pygvADNorm[i])), 2);
                }
                driftDist = Math.sqrt(driftDist);
                driftDist = driftDist * (1/Math.sqrt(2));
                totalDist[k] = totalDist[k] + driftDist;
            }
            totalDist[k] = totalDist[k] / allPossibleInstances.size();
        }

        // Return the most accurate distance
        for (int i = 0; i < modelBD.switchPoints.length; i++) {
            if (totalDist[i] < modelBD.switchPoints[i]) return totalDist[i];
        }
        return totalDist[totalDist.length - 1];
    }

    public static double pygvModelDistance(ClassifierModel modelBD, ClassifierModel modelAD) {
        Instances allPossibleInstances = findIntersectionBetweenInstances(modelBD.allPossibleInstances, modelAD.allPossibleInstances);
        Distance hellinger = new HellingerDistance();
        return hellinger.findPyGvDistance(modelBD, modelAD, allPossibleInstances);
    }

    public static double pvModelDistance(ClassifierModel modelBD, ClassifierModel modelAD) {
        double driftDist = 0.0f;

        // Trim last attribute as allPossibleCombinations contains a class attribute which has NaN
        Instances trimmedBD = trimClass(modelBD.allPossibleInstances);
        Instances trimmedAD = trimClass(modelAD.allPossibleInstances);
        Instances allPossibleInstances = findIntersectionBetweenInstances(trimmedBD, trimmedAD);

        for (Instance combination : allPossibleInstances){
            driftDist = driftDist +
                    Math.pow((Math.sqrt(modelBD.findPv(combination.toDoubleArray()))
                            - Math.sqrt(modelAD.findPv(combination.toDoubleArray()))), 2);
        }
        driftDist = Math.sqrt(driftDist);
        driftDist = driftDist * (1/Math.sqrt(2));
        return driftDist;
    }

    protected static Instances findIntersectionBetweenInstances(Instances instances1, Instances instances2) {
        // Create a hashed mapped set of instances1 first
        HashMap<Integer, Instance> baseMap = new HashMap<>(instances1.size());
        for (Instance instance : instances1) {
            Integer hash = Arrays.hashCode(instance.toDoubleArray());
            if (!baseMap.containsKey(hash)) baseMap.put(hash, instance);
        }

        // Check each instance in instances2, if it is in instances1's baseMap, put into final Instances to return
        Instances finalInstances = new Instances(instances2, instances1.size() + instances2.size());
        for (Instance instance : instances2) {
            Integer hash = Arrays.hashCode(instance.toDoubleArray());
            if (baseMap.containsKey(hash)) finalInstances.add(instance);
        }
        return finalInstances;
    }

    protected static Instances findUnionBetweenInstances(Instances instances1, Instances instances2) {
        Instances allInstance = new Instances(instances1);
        allInstance.addAll(instances2);

        // Create a hashed mapped set of instances1 first
        Instances finalInstances = new Instances(instances1, instances1.size() + instances2.size());
        HashMap<Integer, Instance> baseMap = new HashMap<>(instances1.size());
        for (Instance instance : allInstance) {
            Integer hash = Arrays.hashCode(instance.toDoubleArray());
            if (!baseMap.containsKey(hash)) {
                baseMap.put(hash, instance);
                finalInstances.add(instance);
            }
        }
        return finalInstances;
    }

    protected void generateCombinations(int currentIndex, ArrayList<Double> auxCombination){
        if (currentIndex == this.xPossibleValues.size()){
            Instance inst = new DenseInstance(dataSet.numAttributes());
            inst.setDataset(dataSet);
            for (int j = 0; j < auxCombination.size(); j++) {
                inst.setValue(j, auxCombination.get(j));
            }
            allPossibleInstances.add(inst);
        }
        else {
            ArrayList<String> curValues = this.xPossibleValues.get(currentIndex);
            for (String value : curValues){
                // Add the new possible x value to the list of values and call itself with the new list
                auxCombination.add(Double.parseDouble(value));
                // Add the generated combinations with the new list to the existing combinations
                generateCombinations(currentIndex + 1, auxCombination);
                // Remove the added x value from the list of values
                auxCombination.remove(auxCombination.size() - 1);
            }
        }
    }

    protected void generateSampleCombinations() {
        hashedInstanceSet = new HashMap<>(dataSet.size());

        Instances trimmedData = trimClass(dataSet);

        if (trimmedData.size() != dataSet.size()) throw new RuntimeException();
        for (int i = 0; i < dataSet.size(); i++) {
            Integer hash = Arrays.hashCode(trimmedData.get(i).toDoubleArray());
            if (!hashedInstanceSet.containsKey(hash)) {
                Instance inst = new DenseInstance(1.0, dataSet.get(i).toDoubleArray());
                inst.setDataset(dataSet);
                inst.setClassMissing();
                allPossibleInstances.add(inst);
                hashedInstanceSet.put(hash, inst);
            }
        }
    }
}
