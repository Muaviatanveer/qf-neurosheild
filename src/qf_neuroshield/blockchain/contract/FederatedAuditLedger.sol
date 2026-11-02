// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FederatedAuditLedger
 * @notice Permissioned control-plane ledger for QF-NeuroShield Federated Learning.
 *         Stores cryptographic hashes and provenance metadata only (never raw media or weights).
 */
contract FederatedAuditLedger {
    address public coordinator;
    uint256 public currentRound;
    bytes32 public latestGlobalModelHash;

    struct Participant {
        bool isRegistered;
        string orgName;
    }

    struct UpdateRecord {
        uint256 round;
        address participant;
        bytes32 baseModelHash;
        bytes32 updateHash;
        uint256 timestamp;
        bool approved;
    }

    struct GlobalModelRecord {
        uint256 round;
        bytes32 globalModelHash;
        string aggregationPolicy;
        uint256 timestamp;
    }

    mapping(address => Participant) public participants;
    mapping(uint256 => mapping(address => bool)) public hasSubmittedInRound;
    mapping(uint256 => GlobalModelRecord) public globalModels;
    UpdateRecord[] public updateHistory;

    event ParticipantRegistered(address indexed participant, string orgName);
    event UpdateRecorded(uint256 indexed round, address indexed participant, bytes32 updateHash, bool approved);
    event GlobalModelRecorded(uint256 indexed round, bytes32 globalModelHash, string aggregationPolicy);

    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "Only coordinator can call this function");
        _;
    }

    modifier onlyRegistered(address participant) {
        require(participants[participant].isRegistered, "Participant not registered");
        _;
    }

    constructor(bytes32 initialGlobalModelHash) {
        coordinator = msg.sender;
        currentRound = 0;
        latestGlobalModelHash = initialGlobalModelHash;
        globalModels[0] = GlobalModelRecord(0, initialGlobalModelHash, "INITIAL_W0", block.timestamp);
    }

    function registerParticipant(address participant, string calldata orgName) external onlyCoordinator {
        participants[participant] = Participant(true, orgName);
        emit ParticipantRegistered(participant, orgName);
    }

    function recordUpdate(
        address participant,
        uint256 round,
        bytes32 baseModelHash,
        bytes32 updateHash,
        bool approved
    ) external onlyCoordinator onlyRegistered(participant) {
        require(round == currentRound + 1, "Invalid training round");
        require(baseModelHash == latestGlobalModelHash, "Base model hash mismatch");
        require(!hasSubmittedInRound[round][participant], "Duplicate submission in same round");

        hasSubmittedInRound[round][participant] = true;
        updateHistory.push(UpdateRecord(round, participant, baseModelHash, updateHash, block.timestamp, approved));

        emit UpdateRecorded(round, participant, updateHash, approved);
    }

    function recordGlobalModel(
        uint256 round,
        bytes32 globalModelHash,
        string calldata aggregationPolicy
    ) external onlyCoordinator {
        require(round == currentRound + 1, "Round sequence mismatch");
        currentRound = round;
        latestGlobalModelHash = globalModelHash;
        globalModels[round] = GlobalModelRecord(round, globalModelHash, aggregationPolicy, block.timestamp);

        emit GlobalModelRecorded(round, globalModelHash, aggregationPolicy);
    }
}