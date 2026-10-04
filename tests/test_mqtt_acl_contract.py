from pathlib import Path


ACL = Path("mosquitto/config/acl")


def test_ems_pipeline_can_read_topics_used_by_api_bridge():
    """The shared deployment identity must be able to consume API inputs."""
    text = ACL.read_text()
    block = text.split("user ems_pipeline", 1)[1]
    assert "topic read home/ui/events" in block
    assert "topic read home/plug/+/command" in block
