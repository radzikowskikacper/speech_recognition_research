package com.ips.kradziko.phd.core.data_handler;

import org.javatuples.Sextet;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Created by kradziko on 17/03/21.
 */
public class DataHandler {
    private Map<String, List<Sextet<String, String, Integer, Double, Double, Double>>> data;
    private Map<String, List<String>> fnames;

    public void loadData(String data_dir) throws IOException {
        data = new HashMap<>();
        fnames = new HashMap<>();

        for(String filename : new String[]{"sentence1.tab", "word1.tab"}){//, "sentence3.tab"}){
            BufferedReader br = new BufferedReader(new FileReader(data_dir + "/doc/JEcontent/tab/" + filename));
            String sCurrentLine;

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts = sCurrentLine.split("\\.wav");
                parts[0] = parts[0].trim();
                parts[0] += ".wav";
                parts[parts.length - 1] = parts[parts.length - 1].trim();

                data.putIfAbsent(parts[parts.length - 1], new ArrayList<>());
                fnames.putIfAbsent(parts[parts.length - 1], new ArrayList<>());
                fnames.get(parts[parts.length - 1]).add(parts[0]);
            }
        }

        Files.walk(Paths.get(data_dir + "/lbl")).filter(Files::isRegularFile)
                .forEach(file -> {
                    if(!(file.toString().contains("sentence") || file.toString().contains("word"))
                            || !file.toString().contains("scores")) return;
                    System.out.println(file);
                    String[] parts = file.toString().split("/");
                    int gender = parts[parts.length - 3].contains("female") ? 0 : 1;
                    int pos = 2;
                    if(parts[parts.length - 2].contains("segmental")) pos = 0;
                    else if(parts[parts.length - 2].contains("intonation") || parts[parts.length - 2].contains("accent")) pos = 1;

                    BufferedReader br = null;
                    String sCurrentLine;
                    try {
                        br = new BufferedReader(new FileReader(file.toString()));
                        while((sCurrentLine = br.readLine()) != null) {
                            if(sCurrentLine.charAt(0) == '#') continue;

                            parts = sCurrentLine.split("\\s+");
                            double c = 0;
                            for(int i = parts.length - 1; i > 0; --i) c += Double.valueOf(parts[i]);
                            c /= parts.length - 1;

                            String f = parts[0].split("/")[2];
                            String id = parts[0].split("/")[0] + "/" + parts[0].split("/")[1];

                            loop:
                            for(Map.Entry<String, List<String>> entry : fnames.entrySet()){
                                if(entry.getValue().contains(f)){
                                    for(int i = 0; i < data.get(entry.getKey()).size(); ++i)
                                        if(data.get(entry.getKey()).get(i).getValue0().equals(parts[0]))
                                        {
                                            System.out.println("found");
                                            switch(pos){
                                                case 0: data.get(entry.getKey()).set(i, data.get(entry.getKey()).get(i).setAt3(c)); break;
                                                case 1: data.get(entry.getKey()).set(i, data.get(entry.getKey()).get(i).setAt4(c)); break;
                                                default: data.get(entry.getKey()).set(i, data.get(entry.getKey()).get(i).setAt5(c));
                                            }
                                            break loop;
                                        }

                                    Sextet<String, String, Integer, Double, Double, Double> q =
                                            new Sextet<>(parts[0], id, gender, -1.0, -1.0, -1.0);
                                    switch(pos){
                                        case 0: q = q.setAt3(c); break;
                                        case 1: q = q.setAt4(c); break;
                                        default: q = q.setAt5(c);
                                    }
                                    data.get(entry.getKey()).add(q);
                                    break;
                                }
                            }
                        }
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
    }

    public void transformData_Kaldi(String output_dir, String test_speaker_id) throws IOException {
        Files.deleteIfExists(Paths.get(output_dir));
        new File(output_dir).mkdir();
        new File(output_dir + "/audio/test").mkdirs();
        new File(output_dir + "/audio/train").mkdirs();
        new File(output_dir + "/data/test").mkdirs();
        new File(output_dir + "/data/train").mkdirs();

        //data
    }
}
