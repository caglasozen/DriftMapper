package main.models;

import main.generator.componets.BayesianNetworkGenerator;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import explorer.ChordalysisModelling;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

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

    protected Instances allPossibleInstances;
    protected HashMap<Integer, Instance> hashedInstanceSet;
    protected Instances dataSet;

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
        generateSampleCombinations();

        //Suggest Garbage collector to run
        System.gc();
    }

    abstract double findPygv(String classLabel, String[] xValVector);

    /**
     * Calculate the probability of a certain combination of x values (aka. xVector) occurring
     * @param xValVector The combination of x values (xVector)
     * @return Probability of given vector of x values occurring
     */
    abstract double findPv(String[] xValVector);

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

        Instances allPossibleInstances = modelBD.allPossibleInstances.size() > modelAD.allPossibleInstances.size() ?
                modelBD.allPossibleInstances : modelAD.allPossibleInstances;

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

    public static double pygvModelDistance(SingleClassifierModel modelBD, SingleClassifierModel modelAD) {
        double totalDist = 0.0f;
        WekaToSamoaInstanceConverter converter = new WekaToSamoaInstanceConverter();

        Instances allPossibleInstances = modelBD.allPossibleInstances.size() > modelAD.allPossibleInstances.size() ?
                modelBD.allPossibleInstances : modelAD.allPossibleInstances;

        for (int k = 0; k < allPossibleInstances.size(); k++) {
            Instance inst = allPossibleInstances.get(k);
            double driftDist = 0.0f;
            double[] pygvBD = modelBD.baseClassifier.getVotesForInstance(converter.samoaInstance(inst));
            double[] pygvAD = modelAD.baseClassifier.getVotesForInstance(converter.samoaInstance(inst));
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
            totalDist = totalDist + driftDist;
        }
        totalDist = totalDist / allPossibleInstances.size();

        return totalDist;
    }

    public static double pvModelDistance(ClassifierModel modelBD, ClassifierModel modelAD) {
        double driftDist = 0.0f;

        // Trim last attribute as allPossibleCombinations contains a class attribute which has NaN
        Instances allPossibleInstances = trimClass(modelBD.allPossibleInstances);

        for (Instance combination : allPossibleInstances){
            driftDist = driftDist +
                    Math.pow((Math.sqrt(modelBD.findPv(convertDoubleToStringArray(combination.toDoubleArray())))
                            - Math.sqrt(modelAD.findPv(convertDoubleToStringArray(combination.toDoubleArray())))), 2);
        }
        driftDist = Math.sqrt(driftDist);
        driftDist = driftDist * (1/Math.sqrt(2));
        return driftDist;
    }

    public static String[] convertDoubleToStringArray(double[] list) {
        String[] str_rep = new String[list.length];
        for (int i = 0; i < list.length; i++) {
            str_rep[i] = Double.toString(list[i]);
        }
        return str_rep;
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
